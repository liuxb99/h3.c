import sys, os, glob
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SSD2T/h3c_evidence_20261126/e0d_dump_20261126"

def load(f):
    raw = open(f, 'rb').read()
    if f.endswith('_f32.raw'):
        return np.frombuffer(raw, dtype=np.float32)
    if f.endswith('_f64.raw'):
        return np.frombuffer(raw, dtype=np.float64)
    # bf16: shift into f32 mantissa-exact via uint32
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)

def rowbad(a, cols):
    n = a.size // cols
    b = a[:n * cols].reshape(n, cols)
    bad = (~np.isfinite(b)).sum(axis=1)
    mx = np.array([np.abs(x[np.isfinite(x)]).max() if np.isfinite(x).any() else float('nan') for x in b])
    return bad, mx

ORDER = ["e0_", "e1_", "e2_", "e3_", "s0_", "s1_", "step0_", "video_", "b0_s0_", "b49_s1_"]
for pref in ORDER:
    for f in sorted(glob.glob(f"{D}/{pref}*.raw")):
        a = load(f)
        n_bad = int((~np.isfinite(a)).sum())
        fin = a[np.isfinite(a)]
        mx = float(np.abs(fin).max()) if fin.size else 0.0
        name = os.path.basename(f)
        line = f"{name:48s} n={a.size:>9} bad={n_bad:>8} maxabs_fin={mx:.3e}"
        # row-level detail for 2D-ish tensors
        for cols in (5376, 5120, 96, 1024, 7168, 18432):
            if a.size % cols == 0 and a.size // cols <= 2800 and a.size // cols >= 2:
                bad, mxr = rowbad(a, cols)
                rows = a.size // cols
                nrows_bad = int((bad > 0).sum())
                line += f" rows={rows} rows_bad={nrows_bad} maxrow_bad={int(bad.max())}"
                break
        print(line)
