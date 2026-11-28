#!/usr/bin/env python3
"""P3-C: compare e0k (native model_native_v2) taps vs a2790 gold baseline.
Usage: compare_e0k.py [dir_e0k]  -> prints per-tap nan/inf/maxabs + first divergence."""
import glob, os, sys, numpy as np

EV = "/Volumes/SSD2T/h3c_evidence_20261126"
E0K = sys.argv[1] if len(sys.argv) > 1 else EV + "/e0k_native_v2_20261126"
GOLD = EV + "/a2790_dump_baseline"

def stats_u16(p):
    u = np.fromfile(p, dtype=np.uint16)
    v = (u.astype(np.uint32) << 16).view(np.float32)
    nan = int(np.isnan(v).sum()); inf = int(np.isinf(v).sum())
    fin = v[np.isfinite(v)]
    mx = float(np.abs(fin).max()) if fin.size else 0.0
    return nan, inf, mx, v

rows = []
for p in sorted(glob.glob(E0K + "/*.raw")):
    name = os.path.basename(p)
    g = os.path.join(GOLD, name)
    n, i, mx, v = stats_u16(p)
    if os.path.exists(g):
        gn, gi, gmx, gv = stats_u16(g)
        same_shape = gv.shape == v.shape
        rel = float("nan")
        if same_shape:
            a, b = np.where(np.isfinite(v) & np.isfinite(gv))
            if a.size:
                rel = float(np.sqrt(((v[a]-gv[a])**2).sum()/max(1e-30,(gv[a]**2).sum())))
        rows.append((name, n, i, mx, gn, gi, gmx, rel))
    else:
        rows.append((name, n, i, mx, None, None, None, None))

print(f"{'tap':42s} {'nan':>7s} {'inf':>5s} {'max':>9s} | {'Gnan':>7s} {'Ginf':>5s} {'Gmax':>9s} {'relrms':>8s}")
first_bad = None
for (name, n, i, mx, gn, gi, gmx, rel) in rows:
    bad = n > 0 or i > 0
    gs = f"{gn:7d} {gi:5d} {gmx:9.4g}" if gn is not None else "gold-missing"
    rels = f"{rel:8.4f}" if rel == rel and rel is not None else ("   n/a" if rel is None else "  nan")
    print(f"{name:42s} {n:7d} {i:5d} {mx:9.4g} | {gs} {rels}")
    if bad and first_bad is None:
        first_bad = name
print("\nFIRST NaN/INF tap in e0k:", first_bad)
