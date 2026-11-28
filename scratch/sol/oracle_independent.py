#!/usr/bin/env python3
"""Independent ConvRot int8 quantization oracle.

Re-implements the quantization line from spec WITHOUT importing
scripts/convrot_quant.py, scripts/qkv_layout_gate.py or
scripts/build_int8_convrot.py (no repo module imports at all:
only numpy + stdlib). Streams the 34GB candidate vs the bf16 HF golden.

Spec (fixed): H_gs = h4^{⊗ log4(gs)} / √gs, conference h4 (NOT Sylvester);
symmetric; groups contiguous along K. Line per subject:
  golden bf16 -> rotate each K-group by H_gs (rotation math in bf16:
  f32 accumulate, output rounded to bf16 RNE) -> per-ROW scale =
  max(absmax/127, 1e-30) (divide-then-clamp) -> codes = rint(rot/scale)
  clamp. Compare vs stored I8 codes + F32[N,1] scales.
QKV fused subjects additionally test a kind-grouped row permutation
derived here from shape only (N % 384 == 0, head_dim 128 blocks).

Usage: oracle_independent.py [--subjects a,b,c] [--variant f32|bf16]
       [--out-prefix PATH] [--skip-passthrough] [--selftest]
"""
import argparse, json, math, os, struct, sys, time
import numpy as np

CAND = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
GDIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
CHUNK = 1 << 24  # max elements in flight
FLOOR = np.float32(1e-30)

# ---------- bf16 helpers (RNE) ----------
def bf16_bits_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)

def f32_to_bf16_f32(x):
    """round f32 -> nearest bf16 (RNE), return as f32 values."""
    u = np.ascontiguousarray(x).view(np.uint32)
    u = u + (0x7FFF + ((u >> 16) & 1))
    return (u >> 16 << 16).view(np.float32)

# ---------- Hadamard (conference, own builder) ----------
_H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)

def hadamard_mat(gs):
    q = int(round(math.log(gs, 4)))
    assert 4 ** q == gs
    H = np.ones((1, 1), dtype=np.float32)
    for _ in range(q):
        H = np.kron(H, _H4)
    return H / math.sqrt(gs)

def rotate_group(Xc, q):
    """Xc: (m, gs=4**q) f32 contiguous -> Xc @ H_gs (f32), staged ±1 passes.
    Each of q stages applies h4 along one kron digit axis and /2 (so the
    product carries the /sqrt(gs))."""
    m = Xc.shape[0]
    X = Xc.reshape(m, *[4] * q)
    for ax in range(1, q + 1):
        T = np.moveaxis(X, ax, -1)
        S = T.sum(axis=-1, keepdims=True)
        X = np.moveaxis(S - 2.0 * T[..., ::-1], -1, ax)
        X = np.ascontiguousarray(X) * 0.5
    return X.reshape(m, -1)

def selftest():
    for gs in (64, 256):
        q = int(round(math.log(gs, 4)))
        H = hadamard_mat(gs)
        err = np.abs(H @ H.T - np.eye(gs)).max()
        sym = np.abs(H - H.T).max()
        rng = np.random.default_rng(7)
        X = (rng.standard_normal((17, gs)) * rng.choice([1, 30, -1e-4])).astype(np.float32)
        d = np.abs(rotate_group(X.copy(), q) - X @ H).max()
        step = 1.0 / math.sqrt(gs)
        xmx = float(np.abs(X).max())
        print(f"selftest gs={gs}: orth={err:.2e} sym={sym:.2e} staged-vs-matmul maxdev={d:.2e} = {d/xmx:.2e} of |X|max (f32 partial-sum noise; codes gate is within1)")
        assert err < 1e-5 and sym == 0 and d < 1e-5 * xmx
    print("selftest OK")

# ---------- safetensors hand-parse ----------
import mmap

def open_st(path):
    f = open(path, "rb")
    ln = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(ln))
    return f, hdr, 8 + ln  # DS = 8 + header_len

def pread_np(f, off, n, dtype):
    return np.frombuffer(os.pread(f.fileno(), n, off), dtype)

# ---------- QKV permutation (own derivation) ----------
def qkv_grouped_perm_array(N):
    """Return perm[c] = golden row for kind-grouped file row c.
    Derivation from shape only: N rows = 3 kinds x heads x head_dim(128),
    golden stores per-head interleaved 384-row blocks (head-major, kind
    inside), file stores kind-major [allQ | allK | allV]."""
    assert N % 384 == 0, N
    heads = N // 384
    c = np.arange(N, dtype=np.int64)
    kind = c // (heads * 128)
    head = (c // 128) % heads
    sub = c % 128
    return head * 384 + kind * 128 + sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None, help="comma list filter")
    ap.add_argument("--variant", choices=["f32", "bf16"], default="f32", help="scale math precision")
    ap.add_argument("--out-prefix", default="scratch/sol/oracle")
    ap.add_argument("--skip-passthrough", action="store_true")
    ap.add_argument("--skip-subjects", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        if not args.subjects and args.skip_passthrough:
            return

    t0 = time.time()
    cf, chdr, CDS = open_st(CAND)
    chdr.pop("__metadata__", None)
    idx = json.load(open(os.path.join(GDIR, "model.safetensors.index.json")))["weight_map"]

    gcache = {}
    def golden(name):
        """-> (dtype, shape, shard, hdr_entry, get_bytes(a,b))"""
        shard = idx[name]
        if shard not in gcache:
            gf, ghdr, GDS = open_st(os.path.join(GDIR, shard))
            ghdr.pop("__metadata__", None)
            gcache[shard] = (gf, ghdr, GDS)
        gf, ghdr, GDS = gcache[shard]
        e = ghdr[name]
        return e["dtype"], tuple(e["shape"]), gf, GDS + e["data_offsets"][0]

    # classify candidate keys
    quant, passthru = {}, {}
    for k, v in chdr.items():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
            continue
        if v["dtype"] == "I8":
            quant[k] = v
        else:
            passthru[k] = v
    subjects = sorted(quant)
    if args.subjects:
        want = set(args.subjects.split(","))
        subjects = [s for s in subjects if s in want]
        assert want <= set(subjects), want - set(subjects)
    if getattr(args, "skip_subjects", False):
        subjects = []

    jl = open(f"{args.out_prefix}_subjects.jsonl", "w")
    results = []
    pt_res = {"checked": 0, "byte_identical": 0, "mismatch": []}

    def log(*a):
        print(*a, flush=True)

    for si, subj in enumerate(subjects):
        ce = chdr[subj]
        N, K = ce["shape"]
        with open(CAND, "rb") as f:
            cqs = chdr[subj + ".comfy_quant"]
            f.seek(CDS + cqs["data_offsets"][0])
            side = json.loads(f.read(cqs["data_offsets"][1] - cqs["data_offsets"][0]))
        gs = side["convrot_groupsize"]
        q = int(round(math.log(gs, 4)))
        gdt, gshape, gmm, gbase = golden(subj)
        assert gdt == "BF16" and gshape == (N, K), (subj, gdt, gshape)
        se = chdr[subj + ".weight_scale"]
        assert se["dtype"] == "F32" and se["shape"][0] == N
        saved_scale = pread_np(cf, CDS + se["data_offsets"][0], se["data_offsets"][1] - se["data_offsets"][0], np.float32).reshape(N)

        is_qkv = subj.endswith("attn.qkv_proj.weight")
        perm = perm_full = None
        if is_qkv:
            perm = qkv_grouped_perm_array(N)          # file row -> golden row
            perm_full = np.empty(N, dtype=np.int64)
            perm_full[perm] = np.arange(N)            # golden row -> file row (inverse)

        rpc = max(1, CHUNK // K)
        n_el = exact = within1 = 0
        exact_id = within1_id = 0
        scale_relmax = 0.0
        floor_rows = 0
        floor_bad = 0          # stored floor rows whose candidate codes or golden rows are nonzero
        extra_floor_bad = 0    # rows where my scale==floor but stored != floor
        dot_gf = dot_gi = n_gf = n_gi = n_c = 0.0  # cos accumulators (grouped / identity)
        max_dev = 0
        if is_qkv:
            myq_full = np.empty((N, K), dtype=np.int8)          # pipeline codes indexed by golden row
            scale_full = np.empty(N, dtype=np.float32)          # pipeline scale indexed by golden row
        scale_file = np.empty(N, dtype=np.float32)              # pipeline scale in file order
        for c0 in range(0, N, rpc):
            c1 = min(N, c0 + rpc)
            m = c1 - c0
            cand = pread_np(cf, CDS + ce["data_offsets"][0] + c0 * K, (c1 - c0) * K, np.int8).reshape(m, K)
            # golden span needed for the (possibly permuted) rows
            if is_qkv:
                src = perm[c0:c1]
                gmin, gmax = int(src.min()), int(src.max())
            else:
                src = None; gmin, gmax = c0, c1 - 1
            span = gmax - gmin + 1
            gbits = pread_np(gmm, gbase + gmin * K * 2, span * K * 2, np.uint16).reshape(span, K)
            W = bf16_bits_to_f32(gbits)
            if is_qkv:
                W = W[src - gmin]
            # rotate groups
            R = np.empty((m, K), dtype=np.float32)
            for gi in range(K // gs):
                R[:, gi * gs:(gi + 1) * gs] = rotate_group(W[:, gi * gs:(gi + 1) * gs], q)
            R = f32_to_bf16_f32(R)  # rotation output rounded to bf16
            absmax = np.abs(R).max(axis=1)
            if args.variant == "bf16":
                scale = f32_to_bf16_f32(absmax / np.float32(127.0))  # round scale to bf16 RNE
            else:
                scale = absmax / np.float32(127.0)
            scale = np.maximum(scale, FLOOR)  # divide-then-clamp
            myq = np.clip(np.rint(R / scale[:, None]), -128, 127).astype(np.int8)
            d = np.abs(cand - myq)
            n_el += d.size
            exact += int((d == 0).sum())
            within1 += int((d <= 1).sum())
            max_dev = max(max_dev, int(d.max()))
            # scale comparison (rows where stored scale > floor)
            ss = saved_scale[c0:c1]
            nz = ss > FLOOR
            if nz.any():
                rel = np.abs(scale[nz] - ss[nz]) / ss[nz]
                scale_relmax = max(scale_relmax, float(rel.max()))
            fl = ~nz
            floor_rows += int(fl.sum())
            if fl.any():
                if cand[fl].any() or (W[fl] != 0).any() or (scale[fl] != FLOOR).any():
                    floor_bad += 1
            extra_floor_bad += int(((scale == FLOOR) & (ss < FLOOR)).sum())
            scale_file[c0:c1] = scale
            if is_qkv:
                myq_full[src] = myq
                scale_full[src] = scale
                C = cand.astype(np.float32) * ss[:, None]       # candidate dequant, rotated domain, file order
                dot_gf += float((C * R).sum()); n_gf += float((R * R).sum()); n_c += float((C * C).sum())
            else:
                deq = myq.astype(np.float32) * scale[:, None]
                # both in rotated domain (H orthonormal => cos invariant for the correct pairing)
                dot_gf += float((deq * R).sum()); n_gf += float((R * R).sum()); n_c += float((deq * deq).sum())
            del W, R, cand
        if is_qkv:
            # whole-subject hypothesis tests
            c_full = pread_np(cf, CDS + ce["data_offsets"][0], N * K, np.int8).reshape(N, K)
            di = np.abs(c_full.astype(np.int16) - myq_full.astype(np.int16))
            exact_id = int((di == 0).sum()); within1_id = int((di <= 1).sum())
            del c_full
            def _cos(a, b):
                return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
            scalecorr_identity = _cos(saved_scale, scale_full)  # candidate scale vs pipeline-of-golden-row-c
            scalecorr_grouped = _cos(saved_scale, scale_file)
        else:
            scalecorr_identity = scalecorr_grouped = None
        cos_grouped = dot_gf / math.sqrt(n_gf * n_c) if n_gf > 0 and n_c > 0 else None
        cos_identity = None
        w1 = within1 / n_el
        w1_id = within1_id / n_el if is_qkv else None
        if is_qkv:
            layout = "grouped" if (w1 == 1.0) else ("identity" if w1_id == 1.0 else "FAIL")
        else:
            layout = "identity"
        zero_ok = (floor_bad == 0 and extra_floor_bad == 0)
        res = {"name": subj, "gs": gs, "shape": [N, K], "n_elems": n_el,
               "codes_within1_frac": w1, "codes_exact_frac": exact / n_el,
               "scale_relmax": scale_relmax, "zero_rows_ok": bool(zero_ok),
               "floor_rows": floor_rows, "floor_bad": floor_bad,
               "max_code_dev": max_dev, "layout": layout}
        if is_qkv:
            res.update({"qkv_within1_identity": w1_id, "qkv_exact_identity": exact_id / n_el,
                        "cos_dequant_grouped": cos_grouped,
                        "qkv_scalecorr_grouped": scalecorr_grouped,
                        "qkv_scalecorr_identity": scalecorr_identity})
        else:
            res["cos_dequant_vs_golden"] = cos_grouped
        res["pass"] = bool(w1 == 1.0 and scale_relmax <= 1e-2 and zero_ok and layout in ("grouped", "identity"))
        results.append(res)
        jl.write(json.dumps(res) + "\n"); jl.flush()
        log(f"[{si+1}/{len(subjects)}] {subj} gs={gs} within1={w1:.6f} exact={exact/n_el:.6f} relmax={scale_relmax:.2e} floor={floor_rows} bad={floor_bad} layout={layout} pass={res['pass']} ({time.time()-t0:.0f}s)")

    # pass-through byte compare
    if not args.skip_passthrough:
        for k in sorted(passthru):
            ce = passthru[k]
            ce0, ce1 = ce["data_offsets"]
            gdt, gshape, gf, gbase = golden(k)
            ok = (gdt == ce["dtype"] and tuple(gshape) == tuple(ce["shape"]))
            if ok:
                n = ce1 - ce0
                step = 1 << 24
                for off in range(0, n, step):
                    ln2 = min(n, off + step) - off
                    if os.pread(cf.fileno(), ln2, CDS + ce0 + off) != os.pread(gf.fileno(), ln2, gbase + off):
                        ok = False; break
            pt_res["checked"] += 1
            if ok:
                pt_res["byte_identical"] += 1
            else:
                pt_res["mismatch"].append(k)
        log(f"passthrough: {pt_res['byte_identical']}/{pt_res['checked']} byte-identical; mismatches={pt_res['mismatch'][:8]}")

    qkv = [r for r in results if r["name"].endswith("attn.qkv_proj.weight")]
    qkv_grouped_all = bool(qkv) and all(r["layout"] == "grouped" for r in qkv)
    qkv_identity_max = max((r["qkv_within1_identity"] for r in qkv), default=None)
    checked = len(results)
    passed = sum(r["pass"] for r in results)
    w1min = min((r["codes_within1_frac"] for r in results), default=None)
    relmax = max((r["scale_relmax"] for r in results), default=0.0)
    pt_ok = args.skip_passthrough or (pt_res["byte_identical"] == pt_res["checked"])
    overall = "PASS" if (checked == len(quant) and passed == checked and pt_ok and (not qkv or qkv_grouped_all)) else ("FAIL" if passed < checked or not pt_ok or not qkv_grouped_all else "PARTIAL")
    summary = {"subjects_checked": checked, "subjects_pass": passed,
               "codes_within1_min": w1min, "worst_scale_relerr": relmax,
               "qkv_subjects": len(qkv), "qkv_grouped_match": qkv_grouped_all,
               "qkv_identity_within1_max": qkv_identity_max,
               "passthrough": pt_res, "variant_scale_math": args.variant,
               "verdict": overall, "elapsed_s": round(time.time() - t0, 1),
               "imports": "numpy+stdlib only; no scripts/ imports"}
    out = {"summary": summary, "subjects": results}
    with open(f"{args.out_prefix}_verdict.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
