#!/usr/bin/env python3
# Native A8W8 dump vs bf16-dequant baseline: per-tap deviation scan (chain order).
# Usage: compare_native.py [NATIVE_DIR] [BASE_DIR]
# Verdict lens: rel = max|native-base| / max|base| per tap (value-range relative;
# requant noise on int8-computed taps is expected ~ requant floor, NOT a bug flag;
# the KEY signals are (1) taps present/absent, (2) shape mismatches, (3) rel >> 0.25
# at the FIRST tap that consumes a native int8 GEMM output, (4) NaN/inf.)
import os, sys, numpy as np

NAT = sys.argv[1] if len(sys.argv) > 1 else "/Users/cyliu99/AI/h3c_mac_20261125/evidence/native_dump_20261126"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/Users/cyliu99/AI/h3c_mac_20261125/evidence/fulldit_dump_baseline"

DT = {"bf16": np.uint16, "f32": np.float32, "f64": np.float64, "u32": np.uint32, "i32": np.int32}

def manifest(d):
    m = {}
    with open(os.path.join(d, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            tag, rows, cols, dt, by = p[0], int(p[1]), int(p[2]), p[3], int(p[4])
            m[tag] = (rows, cols, dt, by)
    return m

def load(d, tag, rows, cols, dt, by):
    path = os.path.join(d, f"{tag}_{rows}x{cols}_{dt}.raw")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(d) if f.startswith(tag + "_")]
        if not cands: return None
        path = os.path.join(d, cands[0])
    raw = np.fromfile(path, dtype=DT[dt], count=by // DT[dt]().itemsize)
    if dt == "bf16":
        x = raw.astype(np.uint32) << 16
        return x.view(np.float32)
    if dt == "u32" or dt == "i32":
        return raw.astype(np.int64)
    return raw.astype(np.float64 if dt == "f64" else np.float32).astype(np.float64)

mb, mn = manifest(BASE), manifest(NAT)
only_b = sorted(set(mb) - set(mn)); only_n = sorted(set(mn) - set(mb))
print(f"taps base={len(mb)} native={len(mn)}  only_base={len(only_b)} only_native={len(only_n)}")
for t in only_b[:20]: print("  missing in native:", t)
for t in only_n[:20]: print("  extra in native:", t)

# chain order: step-major, block-major, math order inside block
ORDER = ["hidden_in","adaln_mod_table_0","qkv_pre_norm","q_post_qknorm_rope","k_post_qknorm_rope",
         "v_after_qkv","attention_out","out_proj_out","post_attn_hidden","mlp_adaln_in",
         "fc1_out","swiglu_out","fc2_out","block_out"]
import re
def order_key(tag):
    try:
        return _order_key_inner(tag)
    except Exception:
        return (998, 998, 998, tag)

def _order_key_inner(tag):
    m = re.match(r"^b(\d+)_s(\d+)_", tag)
    if m:
        blk = int(m.group(1)); st = int(m.group(2))
        mid = tag[m.end():]
        pos = next((i for i, name in enumerate(ORDER) if mid.startswith(name)), 99)
        return (st, blk, pos, tag)
    if tag.startswith("s") and "_s" not in tag and len(tag) > 2 and tag[1].isdigit():
        st = int(tag[1]); return (st, 999, 90, tag)
    if tag.startswith("step"):
        return (int(tag[4]), 999, 80, tag)
    if tag.startswith("video_latent"):
        return (99, 999, 0, tag)
    return (0, -2, -3, tag)

print("\n== per-tap deviations (chain order) ==")
first_flag = None
for tag in sorted(set(mb) & set(mn), key=order_key):
    rb, cb, db, yb = mb[tag]; rn, cn, dn, yn = mn[tag]
    if (rb, db) != (rn, dn) or rb*cb != rn*cn:
        print(f"SHAPE/DTYPE DIFF {tag}: base {rb}x{cb}_{db} vs native {rn}x{cn}_{dn}")
        continue
    a = load(BASE, tag, rb, cb, db, yb); b = load(NAT, tag, rn, cn, dn, yn)
    if a.shape != b.shape: b = b.reshape(a.shape)
    bad = (~np.isfinite(b)).sum()
    diff = np.abs(np.nan_to_num(b) - a); maxd = float(diff.max()); maxr = float(np.abs(a).max())
    rel = maxd / max(maxr, 1e-30)
    flag = ""
    if bad: flag = f" NaN/inf={bad}"
    if rel > 0.25 or bad:
        flag += "  <== FLAG"
        if first_flag is None: first_flag = tag
    if tag.startswith(("b0_", "b1_", "b2_", "s0_", "s1_", "step")) or flag:
        print(f"{tag:46s} rel={rel:.3e} maxabs_ref={maxr:.3e}{flag}")
print("\nFIRST FLAGGED TAP:", first_flag)
