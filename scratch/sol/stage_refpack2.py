#!/usr/bin/env python3
"""Ref2VA eager-pack refiner layout check (v2: list candidate keys explicitly)."""
import sys, json, glob, os
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import numpy as np
from measure_lib import open_st, hf_tensor, cos_sim, deq_bf16
from st_scan import grouped_from_interleaved

OUT = open("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results_refpack.jsonl", "w")
def JD(x): return json.dumps(x, ensure_ascii=False)

PACK = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/Ref2VA/transformer"
OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"

shards = sorted(glob.glob(os.path.join(PACK, "*.safetensors")))
OUT.write(JD({"note": "pack shards", "n": len(shards), "names": [os.path.basename(s) for s in shards][:6]}) + "\n")
cand = {}
for p in shards:
    st = open_st(p)
    hits = [k for k in st["hdr"] if "token_refiner" in k and "attn.qkv_proj.weight" in k]
    if hits:
        cand.setdefault(os.path.basename(p), hits)
    st["f"].close()
OUT.write(JD({"note": "refiner qkv keys", "cand": cand}) + "\n")

stO = open_st(OFFICIAL)
seen = set()
for p in shards:
    st = open_st(p)
    for name in st["hdr"]:
        if not ("token_refiner" in name and "attn.qkv_proj.weight" in name and name not in seen):
            continue
        seen.add(name)
        dtype = st["hdr"][name]["dtype"]
        if dtype != "BF16":
            OUT.write(JD({"key": name, "skip": "dtype " + dtype}) + "\n"); continue
        R = deq_bf16(st, name).astype(np.float32)
        rec = {"key": name}
        if name in stO["hdr"]:
            odt = stO["hdr"][name]["dtype"]
            if odt == "BF16":
                X = deq_bf16(stO, name).astype(np.float32)
                rec["official_vs_pack_cos"] = round(cos_sim(X, R), 5)
            else:
                rec["official_dtype_skip"] = odt
        try:
            H = hf_tensor(name)
            rec["pack_vs_fl2vahf_identity_cos"] = round(cos_sim(R, H.astype(np.float32)), 5)
            rec["pack_vs_fl2vahf_G_cos"] = round(cos_sim(R, grouped_from_interleaved(H.astype(np.float32))), 5)
        except KeyError as exc:
            rec["fl2vahf_lookup_err"] = str(exc)[:80]
        OUT.write(JD(rec) + "\n")
    st["f"].close()
stO["f"].close()
OUT.close()
print("REFPACK2_DONE")
