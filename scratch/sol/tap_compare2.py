#!/usr/bin/env python3
"""P3-C tap-by-tarm comparison v2.

Usage: tap_compare2.py DIR_A DIR_B OUT [max_step]
Reads manifest.tsv + <tap>_<rows>x<cols>_<dtype>.raw files from both dumps.
Compares taps in DIR_A manifest order (= execution order). For each tap:
  - per-side NaN/inf counts (NaN injection point detection)
  - first differing element index (NaN==NaN treated EQUAL)
  - max abs diff, rel rms, count of differing elements
Prints the first N divergences plus a summary. Taps only in one side are listed.
"""
import csv, glob, os, sys
import numpy as np

def load_index(d):
    idx = {}
    with open(os.path.join(d, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            tap, rows, cols, dtype = p[0], int(p[1]), int(p[2]), p[3]
            g = glob.glob(os.path.join(d, f"{tap}_*_{dtype}.raw"))
            if g:
                idx[tap] = (g[0], dtype)
    return idx

def load_arr(path, dtype):
    if dtype == "f32":
        return np.fromfile(path, dtype=np.float32)
    if dtype == "f64":
        return np.fromfile(path, dtype=np.float64)
    if dtype == "i32":
        return np.fromfile(path, dtype=np.int32).astype(np.float64)
    u = np.fromfile(path, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)

def main():
    A_DIR, B_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    max_step = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    ia, ib = load_index(A_DIR), load_index(B_DIR)
    taps = [t for t in ia if t in ib]
    only_a = sorted(set(ia) - set(ib)); only_b = sorted(set(ib) - set(ia))
    # execution order = manifest order of A
    order = []
    with open(os.path.join(A_DIR, "manifest.tsv")) as f:
        for line in f:
            t = line.rstrip("\n").split("\t")[0]
            if t in ia and t in ib: order.append(t)
    out = open(OUT, "w")
    def log(s): print(s, flush=True); out.write(s + "\n"); out.flush()
    log(f"A={A_DIR}\nB={B_DIR}\ncommon taps={len(taps)} only_a={len(only_a)} only_b={len(only_b)}")
    n_ident = 0; n_diff = 0; n_nan_first = None; shown = 0
    for tap in order:
        pa, da = ia[tap]; pb, db = ib[tap]
        floats = {"bf16", "f32", "f64"}
        bridge = da != db and da in floats and db in floats
        if da != db and not bridge:
            log(f"SKIP dtype {tap}: {da} vs {db}"); continue
        a = load_arr(pa, da); b = load_arr(pb, db)
        if bridge:
            a = a.astype(np.float64); b = b.astype(np.float64)
            da = db = "bridge"
        if a.shape != b.shape:
            log(f"SKIP shape {tap}: {a.shape} vs {b.shape}"); continue
        if da == "bf16" or da == "f32" or da == "f64":
            na = int(np.isnan(a).sum() + np.isinf(a).sum())
            nb = int(np.isnan(b).sum() + np.isinf(b).sum())
            if n_nan_first is None and (na or nb):
                n_nan_first = (tap, "A" if na else "B", na, nb)
                log(f"FIRST-NAN tap={tap} side={'A' if na else 'B'} badA={na}/{a.size} badB={nb}/{b.size}")
            eq = (a == b) | (np.isnan(a) & np.isnan(b))
            diff = ~eq
            nd = int(diff.sum())
            if nd == 0:
                n_ident += 1; continue
            n_diff += 1
            fi = int(np.argmax(diff))
            af, bf = a[fi], b[fi]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                mad = float(np.abs(a[finite] - b[finite]).max())
                denom = float(np.sqrt((a[finite]**2).mean())) or 1.0
                rel = mad / denom
            else:
                mad, rel = float("nan"), float("nan")
            if shown < 15:
                shown += 1
                row = fi // b.shape[1] if b.ndim > 1 else -1
                log(f"DIFF tap={tap} n={a.size} badA={na} badB={nb} first_idx={fi} (row={row}) a={af} b={bf} maxabs={mad:.6g} relrms={rel:.6g} ndiff={nd}")
        else:
            n_ident += 1 if np.array_equal(a, b) else 0
    log(f"SUMMARY identical={n_ident} differing={n_diff}")
    log(f"NAN_FIRST {n_nan_first}")
    log(f"TAPS only_a={only_a[:20]}")
    log(f"TAPS only_b={only_b[:20]}")
    log("TAPCMP2_DONE")

main()
