#!/usr/bin/env python3
"""REF2VA eager-pack refiner qkv kind-order vs FL2VA HF shards (official-convrot contradiction check)."""
import sys, json
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import numpy as np
from measure_lib import open_st, hf_tensor, cos_sim, deq_bf16
from st_scan import grouped_from_interleaved

OUT = open("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results_refpack.jsonl", "w")
def JD(x): return json.dumps(x, ensure_ascii=False)

PACK = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/Ref2VA/transformer"
OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"

# locate refiner qkv in the Ref2VA transformer pack (shard glob via index or direct open)
import glob, os
shards = sorted(glob.glob(os.path.join(PACK, "*.safetensors")))
keys = {}
for p in shards:
    st = open_st(p)
    for k in st["hdr"]:
        if "token_refiner" in k and "qkv_proj.weight" in k:
            keys[k] = p
    st["f"].close()

stO = open_st(OFFICIAL)
for name in sorted(keys):
    st = open_st(keys[name])
    R_pack = deq_bf16(st, name).astype(np.float32)
    st["f"].close()
    try:
        R_hf = hf_tensor(name)
    except KeyError:
        R_hf = None
    stO["f"].seek(0)
    X = deq_bf16(stO, name).astype(np.float32) if name in stO["hdr"] else None
    rec = {"key": name, "pack_dtype": st["hdr"][name]["dtype"],
           "official_dtype": stO["hdr"][name]["dtype"] if name in stO["hdr"] else None}
    if X is not None:
        rec["official_vs_pack_cos"] = round(cos_sim(X, R_pack), 5)
        rec["official_vs_pack_idfrac"] = round(float(np.mean([int(np.argmax(np.abs(X[i*128:(i+1)*128]).reshape(-1)) == i) for i in range(0, R_pack.shape[0]//128)])), 4) if False else None
    # is pack stored grouped vs interleaved? compare pack to its own transforms
    rec["pack_is_identity_cos"] = round(cos_sim(R_pack, R_pack), 5)
    rec["vs_G_cos"] = round(cos_sim(R_pack, grouped_from_interleaved(R_pack)), 5)
    rec["multiset_official_vs_pack"] = int(np.array_equal(np.sort(np.asarray(X).view(np.uint16).astype(np.uint32)) if X is not None else [0], np.sort(np.asarray(R_pack).view(np.uint16).astype(np.uint32)))) if X is not None else None
    if R_hf is not None and X is not None:
        rec["official_vs_fl2vahf_cos"] = round(cos_sim(X, R_hf), 5)
    OUT.write(JD(rec) + "\n")
stO["f"].close()
OUT.close()
print("REFPACK_DONE")
