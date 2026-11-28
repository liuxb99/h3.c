#!/usr/bin/env python3
"""Regression probe: today's eager tiny taps (p3f_tA_subtap) vs Nov-26 eager golden (fulldit_dump_baseline).
If layout-invariant block taps are identical, HEAD binary has NOT changed eager semantics.
usage: tapcmp_eager_regression.py GOLD_DIR EAGER_DIR [name_filter_glob]
"""
import sys, os, glob, math
import torch

def load_manifest(d):
    m = {}
    p = os.path.join(d, "manifest.tsv")
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5: continue
            name, rows, cols, dtype, size = parts[0], int(parts[1]), int(parts[2]), parts[3], int(parts[4])
            m[name] = (rows, cols, dtype, size)
    return m

DT = {"bf16": torch.bfloat16, "f32": torch.float32, "f16": torch.float16,
      "f64": torch.float64, "u8": torch.uint8, "i32": torch.int32,
      "i64": torch.int64, "u32": torch.uint32, "u16": torch.uint16, "i8": torch.int8}

def tap_tensor(d, name, meta):
    rows, cols, dtype, size = meta
    path = os.path.join(d, name + ".raw")
    if not os.path.exists(path):
        cand = glob.glob(os.path.join(d, name + "*"))
        if not cand: return None
        path = cand[0]
    t = DT.get(dtype)
    if t is None: return None
    raw = open(path, "rb").read(size)
    if dtype == "bf16":
        v = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
    elif dtype == "f16":
        v = torch.frombuffer(bytearray(raw), dtype=torch.float16)
    else:
        v = torch.frombuffer(bytearray(raw), dtype=t)
    n = rows * cols
    if v.numel() >= n: v = v[:n]
    else: v = torch.cat([v, torch.zeros(n - v.numel(), dtype=v.dtype)])
    try:
        return v.view(rows, cols)
    except Exception:
        return v

INVARIANT_HINTS = ("hidden_in", "block_out", "post_attn", "out_proj_out", "swiglu", "fc2_out",
                   "after_cond", "after_patch", "rope_cos", "rope_sin", "velocity",
                   "final_video_norm", "hidden_out", "cond_hidden", "x0", "latent")

def main():
    GOLD, EAGER = sys.argv[1], sys.argv[2]
    mg, me = load_manifest(GOLD), load_manifest(EAGER)
    shared = sorted(set(mg) & set(me))
    filt = sys.argv[3] if len(sys.argv) > 3 else None
    rows_report = []
    exact = same = 0
    for name in shared:
        if name.startswith("rb"):  # subtap-only today
            continue
        if filt and filt not in name: continue
        a, b = tap_tensor(GOLD, name, mg[name]), tap_tensor(EAGER, name, me[name])
        if a is None or b is None: continue
        if a.shape != b.shape:
            rows_report.append((name, "SHAPE", tuple(mg[name][:2]), tuple(me[name][:2]),
                                "INV" if any(h in name for h in INVARIANT_HINTS) else "lay"))
            continue
        an, bn = a.double(), b.double()
        nan_a, nan_b = int(torch.isnan(an).sum()), int(torch.isnan(bn).sum())
        if nan_a or nan_b:
            rows_report.append((name, "NAN", nan_a, nan_b)); continue
        ident = bool(torch.equal(an, bn))
        inv = any(h in name for h in INVARIANT_HINTS)
        if ident: exact += 1; continue
        d = (an - bn).abs()
        denom = an.abs().max().item()
        relr = d.max().item() / denom if denom > 0 else 0.0
        relrms = (d.pow(2).mean().sqrt() / (an.pow(2).mean().sqrt() + 1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(an.flatten(), bn.flatten(), dim=0).item()
        rows_report.append((name, "DIFF", round(relr, 4), round(relrms, 4), round(cos, 6), "INV" if inv else "lay", int((d > 0).sum())))
        same += 1
    inv_bad = [r for r in rows_report if r[-2] == "INV" and r[1] != "NAN" and (r[4] < 0.999 or r[3] > 0.05)]
    print(f"shared_compared={len(rows_report)+exact} bit_identical={exact} diff={same}")
    print(f"invariant_divergences={len(inv_bad)}")
    rows_report.sort(key=lambda r: (r[1] == 'NAN', -(1 - (r[4] if len(r)>4 and isinstance(r[4], float) else 1))))
    for r in rows_report[:40]: print("\t".join(str(x) for x in r))

if __name__ == "__main__":
    main()
