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
the stored row is `w_rot = w_true · H/√G` with H the 256-wide Sylvester
Hadamard matrix over column groups.  An activation row `a` satisfies

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

* `h3_shaders.metal: h3_rotate_quantize_bf16_int8_rows` — two-pass
  FWLT-ordered threadgroup butterfly per 256-column group (Sylvester
  recursion ⇒ every lane touches every value), 1/√G folded, per-group
  16-lane absmax reduced across 16 lane-groups, symmetric int8 quantize
  with `scale = max(absmax, 1e-8)/127`; dead-block lanes are excluded from
  the row absmax.  Threadgroup = 192 threads (192 chunks = 24 groups of
  8 float4), so it also serves the fallback 192-chunk quantizer shape.
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
