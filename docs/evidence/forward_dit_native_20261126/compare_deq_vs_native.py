#!/usr/bin/env python3
"""P3-C verdict: dequant-arm (native OFF) dump vs native-arm dump, same recipe.
Reports, in chain order, first tap where EITHER arm has NaN/inf and where they diverge."""
import numpy as np, os, re, sys

DEQ = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/dequant_dump_20261126"
NAT = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/native_dump_20261126"

DTYPES = {
    "bf16": np.uint16, "f32": np.float32, "f64": np.float64,
    "u32": np.uint32, "i8": np.int8, "u8": np.uint8,
}

def load(d, tap):
    best = None
    for fn in os.listdir(d):
        if fn.startswith(tap + "_") and fn.endswith(".raw"):
            parts = fn[len(tap)+1:-4].split("_")
            shape = tuple(int(x) for x in parts[:-1]); dt = parts[-1]
            if dt not in DTYPES: continue
            a = np.fromfile(os.path.join(d, fn), dtype=DTYPES[dt])
            if a.size == int(np.prod(shape)):
                a = a.reshape(shape)
                if dt == "bf16":
                    a = (a.astype(np.uint32) << 16).view(np.float32)
                elif dt in ("u32", "u8", "i8"):
                    a = a.astype(np.int64)
                return a, dt
    return None, None

def stats(a):
    if a is None: return None
    fin = np.isfinite(a)
    return dict(nan=int(np.isnan(a).sum()), inf=int(np.isinf(a).sum()),
                absmax=float(np.nanmax(np.abs(a))) if fin.any() else None,
                frac_bad=float(1 - fin.mean()))

def order_key(t):
    m = re.match(r"b(\d+)_s(\d+)_(.+)", t)
    if not m:
        m = re.match(r"s(\d+)_(.+)", t)
        return (9000, int(m.group(1)), 999, m.group(2)) if m else (9999, 0, 0, t)
    blk, st, tail = int(m.group(1)), int(m.group(2)), m.group(3)
    stage = {"hidden_in":0,"qkv_w":1,"qkv_pre_norm":2,"q_norm_w":3,"k_norm_w":4,
             "q_post_qknorm_rope":5,"k_post_qknorm_rope":6,"v_after_qkv":7,
             "attention_out":8,"out_w":9,"out_proj_out":10,"post_attn_hidden":11,
             "adaln_mod_table":12,"mlp_adaln_in":13,"fc1_w":14,"fc1_out":15,
             "swiglu_out":16,"fc2_w":17,"fc2_out":18,"block_out":19}.get(tail, 50)
    return (blk, st, stage, tail)

taps = sorted(set(f.split("_s")[0] for f in os.listdir(DEQ) if f.endswith(".raw")))
names_deq = set()
for fn in os.listdir(DEQ):
    if fn.endswith(".raw"):
        names_deq.add(fn.rsplit("_", 2)[0] if False else None)
# collect tap names properly: tap = everything before _<dims>_<dtype>.raw
def tapname(fn):
    m = re.match(r"(.+)_\d+(?:x\d+)*_\w+\.raw$|^(\w+?)(?:_.+)?\.raw$".replace("\\", ""), fn)
    m = re.match(r"(.+?)_(?:(\d+x)+)(bf16|f32|f64|u32|i8|u8)\.raw$", fn)
    return m.group(1) if m else None

names = sorted({t for t in (tapname(f) for f in os.listdir(DEQ)) if t})
names_n = {t for t in (tapname(f) for f in os.listdir(NAT)) if t}
common = sorted(set(names) & names_n, key=order_key)
print(f"deq taps={len(names)} nat taps={len(names_n)} common={len(common)}")
print(f"only_deq={sorted(set(names)-names_n)[:10]} only_nat={sorted(names_n-set(names))[:10]}")

first_flag = None
rows = []
for tap in common:
    a, dta = load(DEQ, tap); b, dtb = load(NAT, tap)
    sa, sb = stats(a), stats(b)
    bad_a, bad_b = sa["nan"]+sa["inf"] > 0, sb["nan"]+sb["inf"] > 0
    rel = None
    if a is not None and b is not None and a.shape == b.shape:
        af, bf = np.where(np.isfinite(a), a, 0), np.where(np.isfinite(b), b, 0)
        denom = max(float(np.nanmax(np.abs(af))), 1e-30)
        rel = float(np.max(np.abs(af-bf)) / denom)
    flag = bad_a or bad_b or (rel is not None and rel > 0.5)
    rows.append((tap, sa, sb, rel, flag))
    if flag and first_flag is None:
        first_flag = tap

print("\n== first 12 chain-ordered taps (deq vs native) ==")
shown = 0
for tap, sa, sb, rel, flag in rows:
    if shown >= 12: break
    rels = f"{rel:.3e}" if rel is not None else "shape-diff"
    print(f"{tap:40s} deq: nan={sa['nan']:>9d} inf={sa['inf']:>6d} max={sa['absmax'] if sa['absmax'] is not None else 'NA':<8} | nat: nan={sb['nan']:>9d} inf={sb['inf']:>6d} max={sb['absmax'] if sb['absmax'] is not None else 'NA':<8} | maxdiffrel={rels} {'<== FLAG' if flag else ''}")
    shown += 1

nflag = sum(1 for r in rows if r[4])
nan_deq = sum(1 for r in rows if r[1]["nan"]+r[1]["inf"] > 0)
nan_nat = sum(1 for r in rows if r[2]["nan"]+r[2]["inf"] > 0)
print(f"\nFLAG taps: {nflag}/{len(rows)} | deq NaN-taps: {nan_deq} | native NaN-taps: {nan_nat}")
print(f"FIRST FLAG (chain order): {first_flag}")
