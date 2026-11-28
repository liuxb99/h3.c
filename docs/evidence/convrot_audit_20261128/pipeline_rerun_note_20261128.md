# Chain rerun & gate repair — 2026-11-28 (10:30)

## Chain result (pipeline_status.json, chain_done=true)
- parity ✓ | build1 rc=0 | tensor_sha_build1 rc=0 (1035/1035 hashed) | delete1 ✓ | build2 rc=0 | tensor_sha_build2 rc=0 | **repro_compare rc=0 (build1==build2 bitwise — determinism proven)**
- Artifact: /Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors
  size 34,038,912,128 B | sha256 dd098ba96263fd29a75a11b249688e696cbf523007260f61aa9e78d8711ffeeb
- build_manifest.json written; official old file deleted by delete1; free at end 311.1GB
- chain halted only at gate_built_vs_hf (see below); LOAD gate correct fail-closed (no .ready marker yet)

## Gate failure = tool drift, NOT content (FACT, pinned by evidence)
1. ImportError: qkv_layout_gate.py imports rotate_stages_f32 + qkv_grouped_perm, absent from convrot_quant.py (mid-refactor removal; no definitions anywhere incl. engine_backup_prepatch).
2. After restoring helpers, KeyError 'blocks.0.adaln_proj.linear.comfy_quant': gate assumed DRAFT sidecar convention (sidecar stem WITHOUT ".weight"); built/official files use VERBATIM append: full subject key + ".weight_scale" / + ".comfy_quant" (pinned from built header: 250/250 `X.weight.weight_scale`, 250/250 `X.weight.comfy_quant`; scale_f32 only "worked" by stem-rebuild coincidence).
3. Quantized subject names are X.weight (I8) ✓ 250 subjects + 285 pass-through ✓ header keys 1036.

## Scripts/ drift fixes (additive or convention-alignment; NO build-path or product change — pending user ratification)
- scripts/convrot_quant.py: ADDED rotate_stages_f32 (f32 group Hadamard == rotate_weight math) + qkv_grouped_perm(rows) 1-arg twin of build_int8_convrot.py local (semantics pinned by probe_qkv_layout empirical result). Existing functions untouched → built artifact semantics unchanged.
- scripts/qkv_layout_gate.py: scale_f32/groupsize now ent(name+suffix) on FULL subject key; quant_subs derivation uses full subject key. (3 hunks total.)
- Sanity of restored helpers: rotate∘rotate == identity (max |Δ| 3.1e-6 f32); qkv perm is a permutation, torch roundtrip exact.
- These are gate-side only: engine/builder untouched; official verbatim format untouched (loader unaffected).

## Rerun2 (10:26-10:43) — verdict FAIL but ALL FOUR failures were gate bugs, not content
Evidence: qkv_layout grouped cos min 0.9999527 (50/50), pass_through 285/285, quant_cos median 0.999948;
the 200 relrms values match 1-1/gs (0.9375 gs=256 family, 0.875) to 4 decimals — smoking gun.
Gate bugs found & fixed (all tool-side):
1. dequant_chunk applied an EXTRA /sqrt(gs) on top of already-normalized rotate_stages_f32 → relrms ≈ 1-1/gs while cos (scale-invariant) stayed 0.9999. Line 134 I8 path had it right; the two dequant paths in one file were mutually inconsistent.
2. classify_permutation compared flat codes_i8 (1-D) against (N,K) requant → always MULTISET_SHAPE tag; comparison never executed. Fixed: reshape to rc.shape before per-row sort.
3. BufferError at close: mmap-exported views (scale, last-iteration slices) outlive frame; close now gc + tolerant (data already verified; cosmetic crash only).
Cos/relrms checks rerun: scratch/sol/chain_gate_built_vs_hf_rerun3.log (proc_758efd80af98). Expected: PASS (relrms → ~code drift level; cos unchanged 0.9999 >> 0.999 min).
- On PASS: add .ready marker + freeze + asset registration, then engine smoke (load via engine at 640-class draft).
- Prior false alarm (mine): watcher grep matched historical "build1 rc=1" lines from run#1 (InterruptedError 00:3x) and run#2 (NameError shim 04:3x) while run#3 was still healthy.
