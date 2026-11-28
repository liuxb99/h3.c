#!/usr/bin/env python3
"""engine_scale_keys.py — header-only probe: which sidecar scale-key convention
does the engine-consumed asset use? (engine = convrot_engine_patched.safetensors)"""
import mmap, json, struct
import scale_ratio_official_vs_built as P

h, ds = P.hdr("/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors")
sc = [k for k in h if k.endswith("_scale")]
print("engine asset scale keys:", len(sc), "sample:", sorted(sc)[:2])
i8 = [k for k, e in h.items() if e["dtype"] == "I8"]
print("engine I8 subjects:", len(i8), "sample:", sorted(i8)[:2])
both_append = sum(1 for k in sc if k.endswith(".weight_scale") and not k.endswith(".weight.weight_scale"))
print("strip-convention spellings:", both_append)
