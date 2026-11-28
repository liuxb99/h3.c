#!/usr/bin/env python3
"""repro_compare.py — build1 vs build2 whole-file + per-tensor sha comparison."""
import json
import os

S = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol"
a = json.load(open(os.path.join(S, "built1_sha.json")))
b = json.load(open(os.path.join(S, "built2_sha.json")))
whole_identical = a["sha256"] == b["sha256"]
per_mismatch = []
for k in sorted(set(a["tensors"]) | set(b["tensors"])):
    ta, tb = a["tensors"].get(k), b["tensors"].get(k)
    if ta != tb:
        per_mismatch.append({"name": k, "b1": (ta or {}).get("sha256"),
                             "b2": (tb or {}).get("sha256")})
out = {"whole_identical": whole_identical,
       "n_tensors": {"b1": a["n_tensors"], "b2": b["n_tensors"]},
       "per_tensor_mismatch_count": len(per_mismatch),
       "per_tensor_mismatch": per_mismatch[:20],
       "sha256_build2": b["sha256"], "sha256_build1": a["sha256"],
       "verdict": "PASS" if whole_identical and not per_mismatch else "FAIL"}
json.dump(out, open(os.path.join(S, "repro_compare.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:1200])
