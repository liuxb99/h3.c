#!/usr/bin/env python3
"""P3-C: three-way tap compare — scalar CPU reference (e0g) vs native (int8)
vs baseline. Scans NaN/inf, rel deviation per tap in chain order.
Pure-numpy; run with the numpy-bearing system python."""
import numpy as np, os, sys, re

def load(path):
    ext = path.rsplit("_", 1)[-1]
    raw = open(path, "rb").read()
    if ext == "bf16":
        a = np.frombuffer(raw, dtype=np.uint16)
        return ((a.astype(np.uint32) << 16)).view(np.float32).copy()
    if ext == "f32":
        return np.frombuffer(raw, dtype=np.float32)
    if ext == "f64":
        return np.frombuffer(raw, dtype=np.float64)
    if ext == "u32":
        return np.frombuffer(raw, dtype=np.uint32)
    if ext == "i8":
        return np.frombuffer(raw, dtype=np.int8)
    return None

def stats(a):
    if a.dtype == np.uint32 or a.dtype == np.int8:
        return dict(n=a.size, bad=0, mx=float(np.max(np.abs(a.astype(np.float64)))), mean=None)
    bad = int(np.count_nonzero(~np.isfinite(a)))
    fin = a[np.isfinite(a)]
    return dict(n=a.size, bad=bad,
                mx=float(np.max(np.abs(fin))) if fin.size else 0.0,
                mean=float(np.mean(np.abs(fin))) if fin.size else 0.0)

def dirname_sort_key(name):
    # chain order: e* first, then block b<n>, then step s<n>, then tap
    m = re.match(r"^b(\d+)_s(\d+)_(.*)$", name)
    if m: return (2, int(m.group(1)), int(m.group(2)), m.group(3))
    m = re.match(r"^e(\d+)_(.*)$", name)
    if m: return (1, int(m.group(1)), 0, m.group(2))
    return (0, 0, 0, name)

def tapname(fname):
    return re.sub(r"_[0-9]+x[0-9]+(_[0-9]+)?_(bf16|f32|f64|u32|i8)\.raw$", "", fname)

def main(dirA, dirB, limit=100000):
    def taps(d):
        out = {}
        for f in os.listdir(d):
            if f.endswith(".raw"):
                out.setdefault(tapname(f), []).append(f)
        return out
    A, B = taps(dirA), taps(dirB)
    common = sorted(set(A) & set(B), key=dirname_sort_key)
    print(f"A={dirA}\nB={dirB}\ncommon taps={len(common)} "
          f"onlyA={len(set(A)-set(B))} onlyB={len(set(B)-set(A))}")
    flags = 0
    for t in common[:limit]:
        fa, fb = sorted(A[t])[0], sorted(B[t])[0]
        if os.path.basename(fa) != os.path.basename(fb):
            print(f"NAME DIFF {t}: {fa} vs {fb}")
            continue
        a, b = load(os.path.join(dirA, fa)), load(os.path.join(dirB, fb))
        if a is None or b is None or a.size != b.size:
            print(f"SIZE DIFF {t}: {a.size if a is not None else '?'} vs "
                  f"{b.size if b is not None else '?'}")
            continue
        sa, sb = stats(a), stats(b)
        badline = ""
        if sa["bad"] or sb["bad"]:
            badline = f" nonfinite: A={sa['bad']} B={sb['bad']}"
        fa_, fb_ = a[np.isfinite(a)] if a.dtype == np.float32 else a, \
                   b[np.isfinite(b)] if b.dtype == np.float32 else b
        if fa_.size == 0 or fb_.size == 0:
            rel = float("nan")
        else:
            denom = max(np.max(np.abs(fa_)), 1e-30)
            rel = float(np.max(np.abs(fa_.astype(np.float64) - fb_.astype(np.float64))) / denom)
        flag = (rel > 0.05) or sa["bad"] > 0 or sb["bad"] > 0
        mark = "  <== FLAG" if flag else ""
        print(f"{t:48s} relA-B={rel:.3e} Amax={sa['mx']:.3e} Bmax={sb['mx']:.3e}{badline}{mark}")
        if flag: flags += 1
        if flags >= 15 and flag and t.startswith("b"):
            print("... (stopped after 15 flags)")
            break

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
