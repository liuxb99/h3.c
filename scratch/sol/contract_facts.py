#!/usr/bin/env python3
"""contract_facts.py — measure the official convrot contract header."""
import json, os, sys, collections
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scripts")
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import st_io as s
from collections import Counter

OFF = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
print("size", os.path.getsize(OFF))
h, f, mm, ds = s.st_open(OFF)
meta = h.get("__metadata__", {})
print("metadata keys:", {k: (v[:60] if isinstance(v, str) else v) for k, v in meta.items()})
dtypes = Counter(v["dtype"] for k, v in h.items() if k != "__metadata__")
print("dtypes:", dict(dtypes))
# scale entries
scales = {k: v for k, v in h.items() if k.endswith(".weight_scale")}
shp = Counter(tuple(v["shape"]) for v in scales.values())
sdt = Counter(v["dtype"] for v in scales.values())
print("scale entries:", len(scales), "shapes:", {str(k): c for k, c in shp.most_common()}, "dtypes:", dict(sdt))
# groupsize inventory from sidecar payloads
gs_inv = Counter()
sidecar_spellings = Counter()
for k, v in h.items():
    if k.endswith(".comfy_quant"):
        sidecar_spellings[k.count(".")] += 0  # placeholder
        cj = json.loads(s.payload(mm, ds, v))
        subj = k[: -len(".comfy_quant")]
        gs_inv[(cj.get("format"), cj.get("convrot"), cj.get("convrot_groupsize"),
                tuple(sorted(cj.keys())))] += 1
        if subj.endswith(".weight"):
            sidecar_spellings["with_weight_suffix"] += 1
        else:
            sidecar_spellings["bare_stem"] += 1
print("sidecar spellings:", dict(sidecar_spellings))
print("groupsize inventory (format,convrot,gs,keys)->n:")
for kk, c in sorted(gs_inv.items(), key=lambda x: str(x[0])):
    print("  ", kk, "->", c)
# sample scale entry shapes for qkv + fc2 + adaln + token_refiner
for probe in ["blocks.0.attn.qkv_proj.weight_scale", "blocks.10.mlp.fc2.weight_scale",
              "blocks.49.adaln_proj.linear.weight_scale",
              "token_refiner.blocks.0.attn.qkv_proj.weight_scale",
              "adaln_single.emb.timestep_embedder.linear_1.weight_scale"]:
    print(probe, "->", h.get(probe, {}).get("shape"), h.get(probe, {}).get("dtype"))
# metadata 'format' entry (full)
for k, v in meta.items():
    print("META", k, "=", (v[:400] if isinstance(v, str) else v))
# main-block qkv row counts (384 multiple check)
import numpy as np
for probe in ["blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.attn.qkv_proj.weight"]:
    if probe in h: print(probe, h[probe]["shape"])
s.st_close(mm, f)
