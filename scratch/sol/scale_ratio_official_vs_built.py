#!/usr/bin/env python3
"""scale_ratio_official_vs_built.py — read-only probe: official sidecar scale vs
h3c-built sidecar scale per subject (median/percentiles of ratio). Decides where
the 0.9375 relrms fingerprint (1 - 1/16 = sqrt(gs)=16 factor) lives: if median
ratio ~16 (or 1/16) the difference is in the STORED scales; if ~1 with some
subjects ~16 it is layout-dependent; if ~1 everywhere the divergence is in the
frozen 00:34 gate bytecode's requant line, not the assets.
CPU only, mmap streaming."""
import mmap, json, struct, sys
import numpy as np

OFF = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
BLT = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"


def hdr(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n = struct.unpack("<Q", mm[:8])[0]
    h = json.loads(mm[8:8 + n])
    ds = 8 + n  # data section starts after u64 + header JSON (incl. padding)
    mm.close(); f.close()
    return h, ds


def rd(path, h, key, ds):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    e = h[key]
    o = e["data_offsets"]
    dtype = {"BF16": np.uint16, "F32": np.float32, "F64": np.float64}[e["dtype"]]
    arr = np.frombuffer(mm, dtype=dtype, count=(o[1] - o[0]) // np.dtype(dtype).itemsize, offset=ds + o[0]).copy()
    mm.close(); f.close()
    if e["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(arr, dtype=np.float64).reshape(e["shape"])


def main():
    ho, dso = hdr(OFF)
    hb, dsb = hdr(BLT)
    # official (strip convention): subject "X.weight" -> scale key "X.weight_scale"
    # built (append convention, 2026-11-28 refactor): "X.weight.weight_scale"
    pairs = []
    for kb, eb in hb.items():
        if not kb.endswith(".weight_scale"):
            continue
        subj = kb[: -len(".weight_scale")]
        ko = subj + "_scale"
        if ko in ho and eb["dtype"] == ho[ko]["dtype"]:
            pairs.append((kb, ko))
    print(f"mapped pairs {len(pairs)}")
    ratios = []
    for kb, ko in sorted(pairs):
        o = rd(OFF, ho, ko, dso).ravel()
        b = rd(BLT, hb, kb, dsb).ravel()
        n = min(len(o), len(b))
        o, b = o[:n], b[:n]
        g = b > 1e-20
        if not g.any():
            continue
        r = o[g] / b[g]
        med = float(np.median(r))
        ratios.append((kb, med))
        if abs(med - 1.0) > 0.02:
            print(f"OFF-UNIT {kb} median={med:.6f} min={r.min():.4f} max={r.max():.4f} n={len(r)}")
    meds = np.array([m for _, m in ratios])
    print(f"subjects measured {len(meds)}: median-of-medians {np.median(meds):.6f}, "
          f"frac within 2% of 1.0 {float((abs(meds-1)<0.02).mean()):.4f}, "
          f"frac >8 {float((meds>8).mean()):.4f}, frac <1/8 {float((meds<0.125).mean()):.4f}")
    worst = sorted(ratios, key=lambda t: -abs(t[1]-1))[:5]
    for k, m in worst:
        print(f"worst {k} median={m:.6f}")


if __name__ == "__main__":
    main()
