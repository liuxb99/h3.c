# Native convrot INT8 (A8W8) weight path — design

Baseline: PR #1 @ 7248ed5 (convrot import + BF16 dequant fallback, kept
intact).  This feature is gated by `H3_NATIVE_INT8=1`; with the variable
unset the engine behaves exactly as the accepted baseline.

## Contract

The MiniMax-H3-comfy-int8 checkpoint stores attention `out_proj` and MLP
`fc2` weights as comfy_quant convrot payloads:

* payload: int8 `{rows, cols}` rows = engine out-dim, cols = engine in-dim
* sidecar: F32 `{rows, 1}` per-row weight scales, `int8_tensorwise`
* quant_info: `group_size=256`, `rotation=int8_convrot_orthogonal`

Semantics (proven by dequant corr=1.0000, `tools/probe_convrot_int8.py`):
the stored row is `w_rot = w_true · H/√G` where H is the 256-wide Kronecker
power `h4⊗4`, `h4[a][b] = -1 iff a XOR b == 3` — the ComfyX convrot_w4a4
Walsh matrix (`_build_hadamard`), NOT the Sylvester/Walsh-Paley 2^k recursion
(corrected 2026-11-26: numerically verified that no input permutation makes
a Sylvester FWLT match the engine's `dit_had()` table; they are different
orthogonal matrices).  The kernel implements H transposed, which is what
`a · w_trueᵀ = (a · H/√G) · w_rotᵀ` needs:

    a · w_trueᵀ = (a · H/√G) · w_rotᵀ

so a consumer that (1) rotates each activation row by the same orthonormal
Hadamard-per-256-group transform, (2) quantizes the rotated row to int8
with per-row dynamic absmax scale, and (3) runs the existing row-scale
int8 GEMM against the *file's* payload and row scales, reproduces the true
product with one quantization step instead of dequant+requant, and with
strictly better outlier behaviour (rotation spreads activation spikes:
post-rotation absmax ≤ pre-rotation absmax, so the quantization step is
finer).

No whole-model dequantization to BF16 is performed for these tensors:
`load_native_int8_pair` binds the int8 payload and the F32 row scales as
engine tensors verbatim (shape/dtype/convrot/group validated at load).

## Scope of the first cut (v0)

Quantized-native: attention out-proj, MLP fc2 (input widths 3584, both
% 256 == 0, so the wrapper's group rule is exact — no padding path yet).
Still BF16 (dequantized at load, unchanged fallback machinery): qkv, fc1,
text encoder, VAE, embeddings/norms.  `int8_qkv` fused producer quantize
is disabled in native mode because the fused RoPE/attention outputs are
unrotated; rotating producers is follow-up work.

## Components

* `h3_shaders.metal: h3_rotate_quantize_bf16_int8_rows` — two-pass fused
  rotation+quantizer per 256-column group.  The transform is applied as
  four base-4 digit XOR-convolutions (`out[a] = v[a]+v[a^1]+v[a^2]-v[a^3]`
  per digit — exactly the Kronecker factorization of `h4⊗4`), digit 0
  in-register on the float4 components, digits 1-2 as `simd_shuffle_xor`
  pairs, digit 3 across the simdgroup pair via threadgroup memory.  The
  orthonormal 1/16 fold lands on the stored SCALE (int8 codes are
  scale-invariant), so `qa · scale_a` reproduces `(H a)/16` bit-for-bit
  against the CPU oracle.  Per-row clipped-absmax/127 scale semantics
  match `h3_quantize_bf16_int8_rows`; dead-block lanes are excluded from
  the row absmax; the wave-loop exchange buffer is parity double-buffered
  (fast-simdgroup next-wave writes must not clobber slow partners'
  same-wave reads); padding rows zero only their own output range.
* `h3_gpu.m` — kernel registered in the pipeline probe list;
  `linear_int8_bf16` layout wrapper routes row-major int8 GEMM inputs
  through `linear_rotq_int8_bf16` when the input's column count is a
  multiple of 256 (the convrot group invariant).  Quantize uses the same
  padded dispatch/numel contract as the plain row quantizer (scale arena
  checks are `>=`-based, verified in `h3_gpu_require_elements`).
* `h3_dit.c` — `native_int8` flag from `H3_NATIVE_INT8`;
  `load_native_int8{,_pair}` binds payload+scale per block (out-proj
  `{HIDDEN,INNER}`, fc2 `{HIDDEN,FFN}`); requantize-at-load passes are
  skipped for the native path; forward MLP uses FC1-BF16 → SwiGLU →
  `linear_int8_bf16`(rotated fc2).  QKV stays on the BF16 branch.

## Validation ladder (each stage vs the accepted baselines)

0. unit (DONE 2026-11-26, `make h3_native_rotquant_tests`):
   * CPU fp64 oracle replicates the kernel op-for-op; the GPU kernel is
     bit-exact on 6 shapes (1/7/128x14336/384x512/2048x7168 rows incl.
     dead-pair wave cols=14336→57-block grid, padding-row path, adversarial
     inputs: +-ulp alternating rows near 1.0 forcing digit-stage
     cancellation, zero/denormal rows exercising the 1/16 fold landing on
     the scale, clip=0.999 tie path) — all bit-exact OK.
   * pairing identity on synthetic convrot layer vs fp64 oracle:
     path A (dequant+bf16 requant) rel_rms 0.0024 (budget 0.015),
     path B (native A8W8) rel_rms 0.0071 (budget 0.05) — PASS;
     discriminator: engine qa dump vs CPU (Ma)/16 oracle rel_rms 0.0069
     ⇒ quantizer+scales verified, remaining delta is bf16 GEMM ordering.
   * oracle-vs-kernel mismatch history that shaped the design: Sylvester
     FWLT ≠ transpose of `dit_had()` table (no permutation reconciles);
     missing wave-parity double-buffering flaked one scale row ~4% off at
     rows=2048 cols=7168 in ~1 of 7 runs (fixed, 4/4 clean after).
1. numeric: 640×352×13f engine dump with `H3_DUMP_ALL=1`, tap-compare
   against the recorded BF16 engine taps — per-block anchor rel errors
   must stay at dequant-requant level or better (rotation should reduce
   fc2/out noise).
2. e2e: 640×352×13f velocity/latent via replay's own velocity (rule:
   never feed engine velocity taps to the scheduler check).
3. perf A/B: 640/544×288×124f/1344×768×124f against the recorded MPS
   baselines (denoise 117s / 504s / 1594s wall; end-to-end 1344 = 36m13s,
   load 205s, TE 103–207s, video VAE 260s, peak ≈ 40.4 GiB) with
   instrumentation off, then memory (peak RSS, swap delta) and cold-vs-hot
   numbers.
4. visual: generated mp4 vs the ComfyX/ComfyUI reference at identical
   model/seed/prompt/recipe.

## Follow-ups (tracked, not in v0)

* fused rotated producers for qkv + fc1 inputs (kills dequant for the
  remaining large tensors; needs RoPE/attention to emit the rotated basis);
* pad-path rotation (input widths not divisible by 256, e.g. text encoder
  tensors) before those layers may go native;
* Apple GPU kernel comparison: native int8 vs fused-dequant vs BF16 GEMM
  timings per stage — the engine's int8 GEMM already sets the baseline
  rate, so gains here come from removing dequant/requant traffic, not from
  a new GEMM; a simdgroup-matrix-int8 rewrite is on the table only if
  measurements justify it (Apple GPUs have no NVIDIA-style int8 tensor
  cores; simdgroup_matrix int8 support must be probed, not assumed);
* freeing the now-unreferenced BF16 out/fc2 copies (≈6.4 GiB) once native
  is proven stable.
