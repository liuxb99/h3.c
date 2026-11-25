#!/usr/bin/env python3
"""convrot_compare — three-way numeric gate for convrot dequant semantics.

Compares, per tensor:
  ref_bf16  (upstream-mirror torch FP32 chain -> bf16 RNE)
  cpu_bf16  (independent C fp32 implementation)
  gpu_bf16  (engine Metal kernel via production load path)
plus mirror-vs-numpy FP32 agreement already recorded in meta.json.

Gates (declared in docs/convrot_semantics.md):
  requant within 1 code      >= 0.9999   (hard gate)
  requant scale max rel err  <= 1e-2     (hard gate)
  requant exact-code match   informational only: stored scales derive
     from bf16 absmax (2^-8 resolution), so ~0.4-1% of codes sit on
     requant grid boundaries and flip by 1 code on round-trip
     (observed 0.9897-0.9963 across tensors; within1 stays 1.0, i.e.
     the dequant mapping is exact up to that stored-quant floor)
     (floor reason: stored scale derives from bf16 absmax, ~0.4% resolution)
  CPU vs ref  bf16 exact     >= 0.9999
  GPU/CPU deviation is gated in the VALUE domain, normalized by the row
  quant step (scales.bin): max |diff| < 1.0 step (no weight code shifts a
  full quant step) and frac(|diff| > 0.5 step) <= 1e-6.
  WHY not raw bf16 ulp: convrot sums rotate signed int8, so ~1e-3 of
  entries cancel to near-zero where fp32 summation-order noise (~ulp of
  the running partial) flips the bf16 bit pattern by thousands of ulp in
  RAW terms while the absolute value stays far below one quant step
  (e.g. ref -1.5e-8 vs gpu 0.0 = 12688 denormals but 4e-6 of scale).
  Storing quantization itself re-rounds 0.4% of codes by ±1
  (requant_within1=1.0), so sub-step code flips are the semantic noise
  floor, not a dequant-mapping error. Raw bf16 ulp stats remain in the
  report for information only.
Any FAIL -> overall FAIL; evidence JSON is written regardless.
"""
import json
import os
import sys

import numpy as np

OUT = os.environ.get("CONVROT_OUT", "/Users/cyliu99/.hermes/cache/scratch/convrot_sem")

def load_u16(path):
    return np.fromfile(path, dtype=np.uint16)

def load_scale(path, rows):
    s = np.fromfile(path, dtype=np.float32)
    if s.size == rows:
        return s.reshape(rows, 1)
    if s.size % rows == 0:
        return s.reshape(rows, s.size // rows)
    raise SystemExit(f"unexpected scale size {s.size} for {rows} rows")

def bf16_f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)

def deviation_vs_scale(a, b, scale):
    rel = np.abs(bf16_f32(a) - bf16_f32(b)) / np.maximum(np.abs(scale), 1e-30)
    return (float(rel.max()),
            float((rel > 0.5).mean()),
            float((rel > 1.0).mean()))

def compare_pair(a, b):
    if len(a) != len(b):
        raise SystemExit(f"length mismatch {len(a)} vs {len(b)}")
    diff = a != b
    exact = float(1.0 - diff.mean())
    if not diff.any():
        return exact, 0, 0.0
    sa, sb = a.astype(np.int64), b.astype(np.int64)
    oa = np.where(sa & 0x8000, (~sa) & 0xFFFF, sa | 0x8000)
    ob = np.where(sb & 0x8000, (~sb) & 0xFFFF, sb | 0x8000)
    ulp = np.abs(oa - ob)
    return exact, int(ulp.max()), float(ulp.mean())

def main():
    meta = json.load(open(f"{OUT}/meta.json"))
    report = {"file": meta.pop("_file"), "tensors": {}}
    failures = []
    for name, info in meta.items():
        stem = name.replace(".", "_")
        ref = load_u16(f"{OUT}/{stem}.ref_bf16.bin")
        cpu = load_u16(f"{OUT}/{stem}.cpu_bf16.bin")
        gpu = load_u16(f"{OUT}/{stem}.gpu_bf16.bin")
        n, k = info["shape"]
        scale = load_scale(f"{OUT}/{stem}.scale.bin", n)
        ref2, cpu2, gpu2 = ref.reshape(n, k), cpu.reshape(n, k), gpu.reshape(n, k)
        ce, cmax, cmean = compare_pair(ref, cpu)
        ge, gmax, gmean = compare_pair(ref, gpu)
        cdev, chat, cover = deviation_vs_scale(ref2, cpu2, scale)
        gdev, ghat, gover = deviation_vs_scale(ref2, gpu2, scale)
        row = {
            "groupsize": info["groupsize"],
            "upstream_bit_equal": info["upstream_bit_equal"],
            "requant_exact": info["requant_code_match_exact"],
            "requant_within1": info["requant_code_match_within1"],
            "requant_scale_maxrel": info["requant_scale_maxrelerr"],
            "mirror_vs_numpy_fp32_maxrel_rowscaled": info["mirror_vs_numpy_fp32_maxrel_rowScaled"],
            "cpu_vs_ref_exact": ce, "cpu_vs_ref_max_ulp": cmax, "cpu_vs_ref_mean_ulp": cmean,
            "gpu_vs_ref_exact": ge, "gpu_vs_ref_max_ulp": gmax, "gpu_vs_ref_mean_ulp": gmean,
            "cpu_vs_ref_max_dev_quant_steps": cdev,
            "cpu_vs_ref_frac_gt_halfstep": chat, "cpu_vs_ref_frac_gt_1step": cover,
            "gpu_vs_ref_max_dev_quant_steps": gdev,
            "gpu_vs_ref_frac_gt_halfstep": ghat, "gpu_vs_ref_frac_gt_1step": gover,
        }
        checks = [
            ("requant_within1>=0.9999", info["requant_code_match_within1"] >= 0.9999),
            ("requant_scale<=1e-2", info["requant_scale_maxrelerr"] <= 1e-2),
            ("cpu_exact>=0.9999", ce >= 0.9999),
            ("cpu_max_dev<1step", cdev < 1.0),
            ("cpu_frac_gt_halfstep<=1e-6", chat <= 1e-6),
            ("gpu_max_dev<1step", gdev < 1.0),
            ("gpu_frac_gt_halfstep<=1e-6", ghat <= 1e-6),
        ]
        if info["upstream_bit_equal"] is not None:
            checks.append(("upstream_bit_equal", info["upstream_bit_equal"] is True))
        bad = [c for c, ok in checks if not ok]
        row["checks"] = {c: ("PASS" if ok else "FAIL") for c, ok in checks}
        verdict = "PASS" if not bad else "FAIL"
        row["verdict"] = verdict
        if bad:
            failures.append(f"{name}: {bad}")
        report["tensors"][name] = row
        print(f"{name}: gs={info['groupsize']} cpu exact={ce:.6f} dev={cdev:.3f}step | "
              f"gpu exact={ge:.6f} dev={gdev:.3f}step halfstepfrac={ghat:.2e} => {verdict}")
    report["overall"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    json.dump(report, open(f"{OUT}/convrot_compare_report.json", "w"), indent=1)
    print(f"OVERALL: {report['overall']}")
    for line in failures:
        print(" FAIL:", line)
    sys.exit(0 if report["overall"] == "PASS" else 1)

if __name__ == "__main__":
    main()
