# P3-C: e0g (Comfy INT8 -> dequant, arm C) vs fulldit_dump_baseline (arm A)
# scan in manifest order; report first divergence + NaN origin. FP64 math.
import sys, os, struct, csv
import numpy as np
A_DIR, B_DIR = sys.argv[1], sys.argv[2]
STEP = sys.argv[3] if len(sys.argv) > 3 else "s0"
def load_bf16(b):
    a = np.frombuffer(b, dtype="<u2").astype(np.uint32)
    return (a << 16).view(np.float32).astype(np.float64)
def load(path, dtype):
    b = open(path, "rb").read()
    n = len(b)
    if dtype == "bf16": v = load_bf16(b)
    elif dtype == "f32": v = np.frombuffer(b, dtype="<f4").astype(np.float64)
    elif dtype == "f64": v = np.frombuffer(b, dtype="<f8")
    elif dtype == "i32": v = np.frombuffer(b, dtype="<i4").astype(np.float64)
    elif dtype == "i64": v = np.frombuffer(b, dtype="<i8").astype(np.float64)
    else: raise ValueError(dtype)
    return v, n
def find_b(tap):
    for fn in os.listdir(B_DIR):
        if fn.startswith(tap + "_") and fn.endswith(".raw"):
            return os.path.join(B_DIR, fn)
    return None
rows = list(csv.reader(open(os.path.join(A_DIR, "manifest.tsv")), delimiter="\t"))
prev_ok = True
stats = []
for r in rows:
    tap, rows_, cols_, dtype, nbytes = r[0], int(r[1]), int(r[2]), r[3], int(r[4])
    # taps live in three naming families: pre-dit (e*/step0_*), step-scoped
    # with "sN_" prefix, and block taps with "_sN_" infix. Compare the s0
    # generation plus all prefix taps; skip s1+ so baseline (1-step) matches.
    short = tap
    if tap.startswith("s0_"):
        short = tap[3:]
    if ("_s0_" not in tap and not tap.startswith("s0_")
            and "_s" not in tap and "s0" not in tap):
        pass  # prefix taps (e0/e1/e3/step0_*) always compared
    if "_s1_" in tap or "_s2_" in tap:
        continue
    a_path = os.path.join(A_DIR, tap + ".bin")
    if not os.path.exists(a_path): continue
    b_path = find_b(short)
    if b_path is None:
        stats.append((tap, "NO_BASELINE", "", "", "")); continue
    try:
        va, na = load(a_path, dtype)
    except Exception as e:
        stats.append((tap, "LOAD_A_ERR", str(e), "", "")); continue
    bname = os.path.basename(b_path)
    bdtype = bname.split("_")[-1][:-4]
    try:
        vb, nb = load(b_path, bdtype)
    except Exception as e:
        stats.append((tap, "LOAD_B_ERR", str(e), "", "")); continue
    if na != nb or va.size != vb.size:
        stats.append((tap, "SIZE", f"{na}", f"{nb}", "")); continue
    va = va[:vb.size]
    nan_a = int(np.isnan(va).sum()); nan_b = int(np.isnan(vb).sum())
    both = ~np.isnan(va) & ~np.isnan(vb)
    if both.sum() and np.any(both):
        d = np.abs(va[both] - vb[both])
        amax = np.abs(vb[both]).max() or 1.0
        maxabs = float(d.max()); rel = maxabs / amax
    else:
        maxabs = rel = float("nan")
    st = "OK" if (nan_a == nan_b and rel < 2e-2) else ("NAN" if (nan_a or nan_b) else "DIFF")
    stats.append((tap, st, f"nanA={nan_a}", f"nanB={nan_b}", f"maxabs={maxabs:.4g} rel={rel:.3g}"))
    if st not in ("OK",) and prev_ok:
        print(f"FIRST-DIVERGENCE >>> {tap}"); prev_ok = False
for tap, st, x, y, z in stats:
    print(f"{st:12s} {tap:45s} {x:12s} {y:12s} {z}")
print("taps compared:", len(stats))
