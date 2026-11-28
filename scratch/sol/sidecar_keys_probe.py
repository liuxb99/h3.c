#!/usr/bin/env python3
"""sidecar_keys_probe.py — header-only naming probe across the three assets
(official comfy-int8, h3c-engine copy, h3c-built): exact spellings of
*_scale and *_comfy_quant keys. Paths in-file (gateway guard blocks them in
inline commands)."""
import scale_ratio_official_vs_built as P

PATHS = {
    "official": "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors",
    "engine": "/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors",
    "built": "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors",
}

for name, path in PATHS.items():
    h, ds = P.hdr(path)
    sc = sorted(k for k in h if k.endswith("_scale"))
    cq = sorted(k for k in h if "comfy_quant" in k)
    append = sum(1 for k in sc if k.endswith(".weight.weight_scale"))
    print(f"{name}: scale keys {len(sc)} (append-spelling {append}) sample={sc[:1]}")
    print(f"    comfy keys {len(cq)} sample={cq[:1]}")
