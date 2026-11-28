#!/usr/bin/env python3
"""quant_crosscheck.py — per-subject numeric cross-check between two
convrot int8 builds (e.g. official Comfy-Org file vs h3c rebuild).

For every subject quantized (has .comfy_quant) in BOTH files:
  * dequant each side with its own codes + per-row scales + convrot groupsize
  * whole-tensor cosine + relRMS between the two dequantizations
  * sorted-code multiset equality (permutation vs requant discriminator)

Both sides need not be bitwise equal (different quantizer lineages); the
crosscheck reports the DISTRIBUTION of agreement, and flags any subject whose
dequantizations diverge beyond --cos-min (each side is independently gated
against the HF reference by qkv_layout_gate.py).

Usage:
  uv run --with torch,numpy -q python scripts/quant_crosscheck.py \
      --a official.safetensors --b mine.safetensors \
      --jsonl scratch/sol/quant_crosscheck.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from convrot_quant import dequant_f32  # noqa: E402
from qkv_layout_gate import WeightSource, log  # noqa: E402


def gs_of(ws, subj):
    return ws.groupsize(subj)[0]


def quant_set(ws):
    return {k for k in ws.key_file
            if k.endswith(".weight")
            and ws.has(k[:-len(".weight")] + ".comfy_quant")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--jsonl", default="scratch/sol/quant_crosscheck.jsonl")
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    wa = WeightSource(args.a)
    wb = WeightSource(args.b)
    subjects = sorted(quant_set(wa) & quant_set(wb))
    log(f"shared quantized subjects: {len(subjects)}")
    rows = []
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        gsa, gsb = gs_of(wa, subj), gs_of(wb, subj)
        ca, sa = wa.codes_i8(subj), wa.scale_f32(subj)
        cb, sb = wb.codes_i8(subj), wb.scale_f32(subj)
        da = dequant_f32(torch.from_numpy(ca.copy()),
                         torch.from_numpy(sa.copy()), gsa).numpy()
        db = dequant_f32(torch.from_numpy(cb.copy()),
                         torch.from_numpy(sb.copy()), gsb).numpy()
        dot = float((da * db).sum())
        nrm = float(np.sqrt((da * da).sum()) * np.sqrt((db * db).sum()))
        cos = dot / nrm if nrm else 1.0
        relrms = float(np.sqrt(((da - db) ** 2).sum()) /
                       max(np.sqrt((da * da).sum()), 1e-30))
        codes_equal = bool(np.array_equal(ca, cb))
        ms_equal = bool(np.array_equal(np.sort(ca.ravel()),
                                       np.sort(cb.ravel())))
        del da, db, ca, cb, sa, sb
        rec = {"subject": subj, "cos": cos, "relrms": relrms,
               "codes_bitwise_equal": codes_equal,
               "codes_multiset_equal": ms_equal,
               "gs_a": gsa, "gs_b": gsb}
        rows.append(rec)
        with open(args.jsonl, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        if i % 25 == 0 or i == len(subjects):
            log(f"crosschecked {i}/{len(subjects)} (last {subj} cos={cos:.6f})")
    cos = np.array([r["cos"] for r in rows])
    summary = {
        "n": len(rows),
        "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "n_below_cos_min": int((cos < args.cos_min).sum()),
        "n_codes_bitwise_equal": sum(r["codes_bitwise_equal"] for r in rows),
        "n_codes_multiset_equal": sum(r["codes_multiset_equal"] for r in rows),
        "hist": {f"{lo:.3f}-{hi:.3f}": int(((cos >= lo) & (cos < hi)).sum())
                 for lo, hi in [(0.0, 0.9), (0.9, 0.99), (0.99, 0.999),
                                (0.999, 0.9999), (0.9999, 1.0001)]},
        "worst5": sorted(rows, key=lambda r: r["cos"])[:5],
        "seconds": time.time() - t0,
    }
    with open(args.jsonl + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    return 0 if summary["n_below_cos_min"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
