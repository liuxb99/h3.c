# R1 — IQ1_S GGUF structural audit (2026-11-28)

Target: `/Volumes/SSD2T/models/MiniMax-H3-gguf/minimax_h3_fl2va_pruned-IQ1_S.gguf`
(HF MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF, `IQ1_S` split, 4 062 866 976 B
= 3.78 GiB, sha256 prefix `76f1d0bcc9052978…`, verified byte-complete.)

Repro: `python3 scripts/iq1s_gguf_audit.py --json-out <out.json>`
(full parse: header, KV, all 532 tensor infos, offset contiguity, per-type byte
census, mapping diff vs the INT8-ConvRot asset header — stdlib only, mmap).

## Verdict: PASS (container contract fully decoded)

- `contiguity_last_end == file size` exactly → tensor set self-consistent.
- offsets are **relative to data_start**; data_start = 45 504 (metadata aligned to 32 B).

## Container facts (FACT, measured)

| item | value |
|---|---|
| GGUF version | 3 |
| n_kv | **0** — quantizer emits zero metadata; no `general.architecture` |
| n_tensors | 532 |
| naming | every key prefixed `model.diffusion_model.` (sd.cpp convention) |

**Type census (sd.cpp ggml fork enum — NOT llama.cpp numbering!):**
FACT from `stable-diffusion.cpp@e239e4b` `ggml/include/ggml.h` (pin 13b9d92):
`IQ1_S = 19`, `IQ1_M = 29`, `BF16 = 30` (llama.cpp says 16/23/24 — never reuse
llama.cpp tables for this file family).

| type | tensors | bytes |
|---|---|---|
| IQ1_S (19) | 208 | 3.645 GiB (96.33 % of file) |
| BF16  (30) | 212 | 0.052 GiB (norms, small vectors) |
| F16   (1)  | 102 | 0.081 GiB (ada/bias payloads) |
| F32   (0)  | 10  | 0.005 GiB |

Measured avg bytes/elem: F32 4.000, F16 2.000, BF16 2.000, **IQ1_S ≈ 0.195**
(≈1.56 bit/wt + per-block scale overhead; sd.cpp rounds each tensor's rows to
whole 256-elem IQ1_S blocks, so per-tensor bpe varies — loader must re-derive
from ggml row-size rules, not from a flat ratio).

## Mapping to h3.c tensor contract (531 of 532 keys match after
`model.diffusion_model.` strip; GGUF dims are reversed-row-major vs safetensors,
480/531 shapes agree directly)

1. **Pruned re-semantics (the whole 51-shape-mismatch set + 1 new key):**
   - `time_embedder.proj_in/proj_out` (4 keys) **removed**; replaced by
     `adaln_t_table [8, 1025] F32` — timestep→embedding baked into an 8-anchor
     table over the 1025-step grid (v-timestep interpolation).
   - every `blocks.N.adaln_proj.linear.weight` shrinks `[96768, 2688] → [8, 96768] F16`
     (8 timestep anchors × 18·5376 modulation). Engine already consumes
     precomputed per-block modulation rows (`h3_dit_schedule_block`,
     `h3_dit.c:2214/2258/2718`) — GGUF form maps onto that path directly.
2. **No sidecars**: the 500 `comfy_quant`/`weight_scale` keys of the INT8 asset
   don't exist here; scales live inside each IQ1_S ggml block.
3. Only GGUF-only key: `adaln_t_table`. Only-in-safetensors weights: exactly the
   4 time_embedder keys above (all else accounted).

## Engine gap list → next rounds

- R2 `h3_gguf.c`: mmap/pread GGUF-v3 container reader mirroring
  `h3_safetensors.c` API (nkv may be 0; tensor infos → same store descriptors).
- R3 IQ1_S **CPU reference** dequant (`scripts/convrot_cpu_ref.c` pattern):
  ggml block_q4_0-like iq1_s layout (scales + 256-elem 1-bit grid-sign codebook);
  parity per-tensor vs ggml `dequantize_q8`-style golden from vendored sd.cpp ggml.
- R4 Metal fused IQ1_S matmul (no full fp16 weight copy): adapt
  `stable-diffusion.cpp ggml/src/ggml-metal/kernels/mul_mv.metal`
  `kernel_mul_mv_iq1_s_f32_impl` + `N_R0_IQ1_S=4` device tuning into
  `h3_shaders.metal`'s GEMV/GEMM entry points; quant-type dispatch in
  `h3_weights.c` (today it hard-gates `int8_tensorwise` + convrot sidecars).
- R5 single-block numeric gate vs sd.cpp golden activations → per-block sweep.
- R6+ video ladder per user recipe (256×256 22f 2-step first).

## Known script limitations (recorded, not blocking)

- audit script's scalar-KV value map is legacy-shifted (GGUF scalars are
  INT32/FLOAT32/BOOLEAN/STRING…); harmless here because nkv = 0, but the real
  loader must implement the GGUF scalar table.
- alignment assumed 32 B (matches this file exactly; loader should read
  `general.alignment` when KV present).
