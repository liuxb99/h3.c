import os, numpy as np
DEQ = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/dequant_dump_20261126"
NAT = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/native_dump_20261126"
DT = {"bf16": np.uint16, "f32": np.float32, "f64": np.float64, "u32": np.uint32, "i32": np.int32}
def manifest(d):
    m = {}
    with open(os.path.join(d, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            m[p[0]] = (int(p[1]), int(p[2]), p[3], int(p[4]))
    return m
def load(d, m, tag):
    if tag not in m: return None
    rows, cols, dt, by = m[tag]
    path = os.path.join(d, f"{tag}_{rows}x{cols}_{dt}.raw")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(d) if f.startswith(tag + "_") and f.endswith(".raw")]
        if not cands: return None
        path = os.path.join(d, cands[0])
    raw = np.fromfile(path, dtype=DT[dt], count=by // DT[dt]().itemsize)
    if dt == "bf16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dt in ("u32", "i32"):
        return raw.astype(np.int64)
    return raw
md, mn = manifest(DEQ), manifest(NAT)
only_d = sorted(set(md) - set(mn)); only_n = sorted(set(mn) - set(md))
print(f"deq taps={len(md)} nat taps={len(mn)} only_deq={len(only_d)} only_nat={len(only_n)}")
print("only_deq sample:", only_d[:8])
print("only_nat sample:", only_n[:8])
for tap in ["b0_s0_hidden_in", "b0_s0_adaln_mod_table", "b0_s0_qkv_pre_norm",
            "b0_s0_attention_out", "b0_s0_post_attn_hidden", "b0_s0_mlp_adaln_in",
            "b0_s0_fc1_out", "b0_s0_fc2_out", "b0_s0_block_out"]:
    parts = [tap.ljust(26)]
    for label, d, m in (("deq", DEQ, md), ("nat", NAT, mn)):
        x = load(d, m, tap)
        if x is None:
            parts.append(f"{label}: MISSING")
            continue
        nan = int(np.isnan(x).sum()); inf = int(np.isinf(x).sum())
        fin = x[np.isfinite(x)]
        mx = float(np.max(np.abs(fin))) if fin.size else float("nan")
        parts.append(f"{label}: nan={nan:8d} inf={inf:5d} max={mx:.3e}")
    print(" | ".join(parts))
