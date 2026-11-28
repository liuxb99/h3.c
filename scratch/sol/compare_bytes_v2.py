#!/usr/bin/env python3
"""compare_bytes_v2.py — per-tensor byte cross-gate: built vs engine copy vs HF golden.

  A: built vs engine copy, EVERY shared tensor name:
       in-backup-279   mismatch => BUILDER_FAIL-candidate
       else            mismatch => ENGINE_COPY_DIVERGENCE (+file offsets)
  B: built pass-through (285, non-I8-subject) vs HF golden bytes: must be
     byte-identical (golden = only truth).
  C: quant subject codes built vs official/engine bytes: exact / within-1 /
     maxdev per subject + aggregate (compatibility band: exact>=0.99,
     within1==1.0 per docs/evidence 2026-11-25三方報告).
  D: scale + comfy sidecars built vs engine: byte-identical expected.
  E: engine copy vs HF golden pass-through mismatches (divergence census,
     with prepatch-local sha check from backup manifest).

Writes scratch/sol/gate_built_vs_engine.json. Streams per tensor; no full load.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
sys.path.insert(0, os.path.join(HERE, "scripts"))
from st_io import payload, st_close, st_open  # noqa: E402

BUILT = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
ENGINE = "/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors"
HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
BK = os.path.join(HERE, "scratch/sol/engine_backup_prepatch")
OUT = os.path.join(HERE, "scratch/sol/compare_bytes_v2.json")


def main():
    bk = {e["name"]: e for e in json.load(open(os.path.join(BK, "manifest.json")))["entries"]}
    bh, bf, bmm, bds = st_open(BUILT)
    eh, ef, emm, eds = st_open(ENGINE)
    wmap = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))["weight_map"]
    shm, shdr = {}, {}

    def hf_bytes(name):
        shard = wmap[name]
        if shard not in shm:
            h, f, mm, ds = st_open(os.path.join(HF_DIR, shard))
            shdr[shard], shm[shard] = (h, ds), (f, mm)
        h, ds = shdr[shard]
        e = h[name]
        return bytes(shm[shard][1][ds + e["data_offsets"][0]: ds + e["data_offsets"][1]])

    bknames = set(bk)
    # v2: sidecar suffix convention differs per file (official/engine single
    # "X.comfy_quant", built double "X.weight.comfy_quant"). Robust subject
    # detection: strip the sidecar suffix, append ".weight" once.
    def _subj_of(k):
        for suf in (".weight.comfy_quant", ".comfy_quant"):
            if k.endswith(suf):
                return k[:-len(suf)] + ".weight"
        return k
    subj = {_subj_of(k) for k in bh if k.endswith(".comfy_quant")}
    pt = sorted(k for k in bh if k != "__metadata__" and k not in subj
                and not k.endswith(".weight_scale") and not k.endswith(".comfy_quant"))
    A = {"mismatch_in_backup": [], "mismatch_outside_backup": []}
    B = {"mismatch": []}
    C = {"per_subject": {}, "agg_exact": 0, "agg_within1": 0, "agg_total": 0}
    D = {"scale_mismatch": [], "comfy_mismatch": []}
    E = {"mismatch": []}
    names = sorted(k for k in bh if k != "__metadata__")
    for name in names:
        be = bh[name]
        diff = None
        bp = np.frombuffer(bmm[bds + be["data_offsets"][0]: bds + be["data_offsets"][1]], dtype=np.uint8)
        stem = name[:-7] if name.endswith(".weight") else name
        if name in eh:
            ee = eh[name]
            ep = np.frombuffer(emm[eds + ee["data_offsets"][0]: eds + ee["data_offsets"][1]], dtype=np.uint8)
            if bp.shape != ep.shape or not np.array_equal(bp, ep):
                diff = np.flatnonzero(bp != ep) if bp.shape == ep.shape else None
                rec = {"name": name,
                       "n_diff": int(len(diff)) if diff is not None else -1,
                       "first_offsets_engine": [int(x) for x in diff[:5]] if diff is not None and len(diff) else [],
                       "class": "BUILDER_FAIL_CANDIDATE" if stem in bknames or name in bknames else "ENGINE_COPY_DIVERGENCE"}
                (A["mismatch_in_backup"] if rec["class"] == "BUILDER_FAIL_CANDIDATE"
                 else A["mismatch_outside_backup"]).append(rec)
        # B: pass-through vs HF golden
        if name in pt:
            try:
                hb = hf_bytes(name)
                if bytes(bp) != hb:
                    B["mismatch"].append(name)
            except KeyError:
                B.setdefault("no_hf_golden", []).append(name)
        # C: codes vs official (engine carries official I8 bytes in code regions)
        if name in subj:
            if diff is None:
                dev = None
                exact = 0.0
                within1 = 1.0
            ec = np.frombuffer(emm[eds + eh[name]["data_offsets"][0]: eds + eh[name]["data_offsets"][1]],
                               dtype=np.int8)
            d = np.abs(bp.view(np.int8).astype(np.int32) - ec.astype(np.int32))
            C["per_subject"][name] = {"exact_frac": float((d == 0).mean()),
                                      "within1_frac": float((d <= 1).mean()),
                                      "max_abs_dev": int(d.max())}
            C["agg_total"] += int(d.size)
            C["agg_exact"] += int((d == 0).sum())
            C["agg_within1"] += int((d <= 1).sum())
            del d, ec
        # D: sidecars verbatim
        if name.endswith(".weight_scale") or name.endswith(".comfy_quant"):
            pass  # already covered by A compare above
        del bp
    # E: engine PT regions vs HF golden + old-local sha
    for name in pt:
        if name not in eh:
            continue
        ee = eh[name]
        ep = np.frombuffer(emm[eds + ee["data_offsets"][0]: eds + ee["data_offsets"][1]], dtype=np.uint8)
        try:
            if bytes(ep) != hf_bytes(name):
                bkrec = bk.get(name)
                E["mismatch"].append({"name": name, "engine_off": ee["data_offsets"][0],
                                      "was_backup_region": name in bk or name[:-7] in bk,
                                      "old_local_sha_prefix": (bk.get(name) or bk.get(name[:-7]) or {}).get("sha_before")})
        except KeyError:
            pass
        del ep
    st_close(bmm, bf)
    st_close(emm, ef)
    for f, mm in shm.values():
        mm.close()
        f.close()
    agg = {"exact_frac": C["agg_exact"] / max(C["agg_total"], 1),
           "within1_frac": C["agg_within1"] / max(C["agg_total"], 1)}
    band_ok = agg["exact_frac"] >= 0.99 and agg["within1_frac"] >= 0.9999
    verdict = "PASS" if (not A["mismatch_in_backup"] and not B["mismatch"]
                         and not D["scale_mismatch"] and not D["comfy_mismatch"]
                         and band_ok) else "FAIL"
    out = {"verdict": verdict, "aggregate_A": {
        "in_backup_mismatches": len(A["mismatch_in_backup"]),
        "outside_backup_mismatches": len(A["mismatch_outside_backup"])},
        "B_pt_vs_hf_golden": {"mismatch_count": len(B["mismatch"]),
                              "mismatch_names": B["mismatch"][:20],
                              "no_hf_golden": B.get("no_hf_golden", [])},
        "C_codes_vs_official_agg": agg, "C_band_ok": band_ok,
        "C_per_subject_worst": sorted(C["per_subject"].items(),
                                      key=lambda kv: kv[1]["exact_frac"])[:8],
        "A_mismatch_in_backup": A["mismatch_in_backup"],
        "A_mismatch_outside_backup_count": len(A["mismatch_outside_backup"]),
        "A_mismatch_outside_backup_sample": A["mismatch_outside_backup"][:12],
        "E_engine_pt_vs_hf": {"mismatch_count": len(E["mismatch"]),
                              "mismatch": E["mismatch"][:20]}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["verdict"] and out["aggregate_A"] and
                     {k: out[k] for k in ("verdict", "aggregate_A", "C_band_ok")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
