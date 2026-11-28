#!/usr/bin/env python3
"""qkv_layout_gate.py — permanent CI gate for convrot-format safetensors
(MiniMax-H3 / h3c engine weight files). Runs on ANY single-file or sharded
convrot weight file against a reference (HF original shards or another convrot
file). Exit code 2 on any violation (0 = pass; --report-only never fails).

Checks:
  [S]  sidecar structure: every I8 weight has F32 [N,1] weight_scale + U8
       comfy_quant {"format":"int8_tensorwise","convrot":true}, groupsize a
       power of two <= 1024 with K % groupsize == 0.
  [S0] zero-scale rows (scale == 1e-30, the adaLN padding rule) must carry
       all-zero codes.
  [P]  pass-through tensors: byte-identical to reference per tensor (dtype +
       shape + sha256 of raw payload). Keys absent from reference: SKIP if
       --allow-missing-reference-keys else FAIL.
  [L]  fused multi-kind projections (name matches qkv|to_qkv|kv_proj):
       row-order probe against BOTH hypotheses derived from the reference
       (reference row order = canonical truth):
         inter   row j = (unit j//3, kind j%3)  per-head interleaved
                 (h3 kernel layout ((row*heads+head)*384) and eager chain)
         grouped stored [allQ][allK][allV]
       best-matching hypothesis must be --expect-layout (default inter) with
       cos >= --cos-min.
  [Q]  quantized tensors numeric gate: dequantize candidate rows via sidecar
       scales + convrot semantics, compare vs reference rows in file order:
       cos >= --cos-min (0.999), relRMS <= --relrms-max (0.05).
       On failure: multiset triage — sorted(u8 codes) of candidate vs sorted
       codes of the reference requantized with the pinned f32 line; mismatch
       fraction < --multiset-tol => tag PERMUTATION (payload multiset
       preserved: row reorder, scale-sidecar-only error, or equivalent
       requant — global flat sort, 2026-11-28 fix: was per-row sort),
       else QUANT_DRIFT.

BF16->F32 is bit-exact (u16 << 16). A reference may be BF16/F32 float or
itself convrot (I8 dequantized with its own scales).

Usage:
  uv run --with torch,numpy -q python scripts/qkv_layout_gate.py \
      --candidate FILE [--reference DIR-or-FILE ...] [--detail-out JSON]
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from convrot_quant import (quant_tensor_f32line, rotate_stages_f32,
                           qkv_grouped_perm)  # noqa: E402
from st_io import payload, st_close, st_open  # noqa: E402

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "U64": 8, "U32": 4, "U16": 2, "BOOL": 1}
QKV_PAT = re.compile(r"(qkv|to_qkv|kv_proj|qkv_proj)(\.weight)?$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class WeightSource:
    """Index one safetensors file or sharded dir; lazy mmap payload access."""

    def __init__(self, path):
        self.files = {}
        if os.path.isdir(path):
            idxp = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(idxp):
                self.wmap = json.load(open(idxp))["weight_map"]
            else:
                singles = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
                assert len(singles) == 1, f"{path}: expected lone safetensors, got {singles}"
                self.wmap = None
                lone_name = singles[0]
            self.dir = path
        else:
            self.dir = None
            self.wmap = None
            path0 = path
        files = (sorted(set(self.wmap.values())) if self.wmap
                 else ([lone_name] if self.dir else [os.path.basename(path0)]))
        self.key_file = {}
        for fn in files:
            p = os.path.join(path, fn) if self.dir else path0
            h, f, mm, ds = st_open(p)
            self.files[fn] = (h, f, mm, ds)
            for k in h:
                if k != "__metadata__":
                    self.key_file[k] = fn
        if self.wmap is None:
            self.wmap = dict(self.key_file)
        self.metadata = {}
        for fn in self.file_order():
            self.metadata.update(self.files[fn][0].get("__metadata__", {}))

    def file_order(self):
        return sorted(self.files)

    def close(self):
        import gc
        gc.collect()
        for fn, (h, f, mm, ds) in self.files.items():
            try:
                st_close(mm, f)
            except BufferError:
                # exported views (e.g. last-iteration slices) still alive;
                # mapping stays valid until they die — cosmetic, not data loss.
                pass

    def ent(self, name):
        fn = self.key_file[name]
        h, f, mm, ds = self.files[fn]
        return fn, h[name], ds

    def has(self, name):
        return name in self.key_file

    def rows_f32(self, name, r0, r1, quant_ctx=None):
        fn, e, ds = self.ent(name)
        mm = self.files[fn][2]
        dtype, shape = e["dtype"], e["shape"]
        K = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        n = r1 - r0
        a = ds + e["data_offsets"][0] + r0 * K * DTYPE_BYTES[dtype]
        buf = mm[a: a + n * K * DTYPE_BYTES[dtype]]
        if dtype == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(n, K).copy()
        if dtype == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32).reshape(n, K).copy()
        if dtype == "F64":
            return np.frombuffer(buf, dtype=np.float64).astype(np.float32).reshape(n, K)
        if dtype == "I8":
            gs, scale_all = quant_ctx
            codes = np.frombuffer(buf, dtype=np.int8).reshape(n, K)
            x = codes.astype(np.float32) * scale_all[r0:r1]
            x = rotate_stages_f32(torch.from_numpy(x.reshape(n, K // gs, gs)))
            return x.numpy().reshape(n, K)
        raise SystemExit(f"unsupported dtype {dtype} in reference {name}")

    def codes_i8(self, name):
        fn, e, ds = self.ent(name)
        mm = self.files[fn][2]
        a, b = e["data_offsets"]
        return np.frombuffer(mm, dtype=np.int8, count=b - a, offset=ds + a)

    def scale_f32(self, name):
        # Official verbatim sidecar convention: sidecar = full subject key
        # + suffix ("X.weight" -> "X.weight.weight_scale" / ".comfy_quant").
        # 2026-11-28: gate previously stripped ".weight" first, matching the
        # old draft convention; the shipped official/built file uses the
        # append form (pinned from the built header: 250/250 keys).
        fn, e, ds = self.ent(name + ".weight_scale")
        mm = self.files[fn][2]
        a, b = e["data_offsets"]
        return np.frombuffer(mm, dtype=np.float32,
                             count=(b - a) // 4, offset=ds + a).reshape(-1, 1).copy()

    def groupsize(self, name):
        fn, e, ds = self.ent(name + ".comfy_quant")
        mm = self.files[fn][2]
        cj = json.loads(payload(mm, ds, e))
        return int(cj["convrot_groupsize"]), cj

    def sha256(self, name):
        fn, e, ds = self.ent(name)
        mm = self.files[fn][2]
        a, b = e["data_offsets"]
        hs = hashlib.sha256()
        step = 1 << 26
        off = ds + a
        while off < ds + b:
            hs.update(mm[off: min(off + step, ds + b)])
            off += step
        return hs.hexdigest()


def requant_ref(ws, ref, name, gs):
    """Requantize the whole reference weight (float or convrot source) with the
    pinned f32 line -> (codes int8 [N,K], scale f32 [N,1])."""
    N = int(ref.ent(name)[1]["shape"][0])
    K = int(np.prod(ref.ent(name)[1]["shape"][1:]))
    ctx = (gs, ref.scale_f32(name)) if ref.ent(name)[1]["dtype"] == "I8" else None
    CH = max(1, min(N, (1 << 24) // max(K, 1)))
    acc = []
    for r0 in range(0, N, CH):
        acc.append(ref.rows_f32(name, r0, min(N, r0 + CH), ctx))
    W = np.concatenate(acc, 0) if len(acc) > 1 else acc[0]
    codes, scale = quant_tensor_f32line(torch.from_numpy(W), gs, mode="int8")
    codes, scale = codes.numpy(), scale.numpy()
    if name.endswith("attn.qkv_proj.weight") and N % 384 == 0:
        codes = qkv_grouped_perm(codes)
        scale = qkv_grouped_perm(scale)
    return codes, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", action="append", default=[],
                    help="reference file or sharded dir; repeatable (first match wins)")
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--relrms-max", type=float, default=0.05)
    ap.add_argument("--expect-layout", default="grouped",
                    choices=["inter", "grouped", "any"],
                    help=("convrot contract stores fused-qkv rows kind-grouped "
                          "(official file + h3c engine rowmap); inter = original "
                          "HF order"))
    ap.add_argument("--allow-missing-reference-keys", action="store_true")
    ap.add_argument("--multiset-tol", type=float, default=0.001)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--detail-out", default=None)
    ap.add_argument("--only", default=None, help="regex filter on tensor names")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cand = WeightSource(args.candidate)
    refs = [WeightSource(p) for p in args.reference]

    def ref_lookup(name):
        for r in refs:
            if r.has(name):
                return r
        return None

    keys = set(cand.key_file)
    quant_subs, sidecar = {}, set()
    for k in keys:
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            sidecar.add(k)
    for k in sorted(keys):
        if k.endswith(".comfy_quant"):
            subj = k[: -len(".comfy_quant")]  # full subject key (e.g. X.weight)
            quant_subs[subj] = cand.groupsize(subj)[0]
    pt_keys = sorted(k for k in keys if k not in quant_subs and k not in sidecar)
    log(f"candidate: {len(quant_subs)} quantized subjects, {len(pt_keys)} pass-through keys, "
        f"sidecar entries {len(sidecar)}")

    failures = []
    details = []
    notes = []
    only = re.compile(args.only) if args.only else None

    def fail(name, check, msg, tag=""):
        failures.append((name, check, msg, tag))
        log(f"FAIL [{check}] {name} {tag} {msg}")

    def classify_permutation(subj, gs, r):
        try:
            if r.ent(subj)[1]["dtype"] == "I8":
                return "REF_I8", float("nan")  # cannot requant an already-quantized reference
            rc, _ = requant_ref(cand, r, subj, gs)
            codes = cand.codes_i8(subj)
            if codes.ndim == 1:  # codes_i8 returns the flat payload
                codes = codes.reshape(rc.shape)
            codes = np.sort(codes.astype(np.uint16).ravel() + 128)
            rcs = np.sort(rc.astype(np.uint16).ravel() + 128)
            if codes.shape != rcs.shape:
                return "MULTISET_SHAPE", float("nan")
            diff = float((codes != rcs).mean())
            return ("PERMUTATION" if diff < args.multiset_tol else "QUANT_DRIFT"), diff
        except SystemExit:
            return "REF_UNQUANTIZABLE", float("nan")

    # ---- [S]/[S0] -----------------------------------------------------------
    n_side = 0
    for subj, gs in sorted(quant_subs.items()):
        if only and not only.search(subj):
            continue
        fn, e, ds = cand.ent(subj)
        ok = True
        if e["dtype"] != "I8":
            fail(subj, "S", f"weight dtype {e['dtype']} != I8")
            ok = False
        if not (e["dtype"] == "I8" and e["dtype"]):
            pass
        N = int(e["shape"][0])
        K = int(np.prod(e["shape"][1:]))
        if gs <= 0 or gs & (gs - 1) or gs > 1024:
            fail(subj, "S", f"groupsize {gs} not a power of two <= 1024")
            ok = False
        elif K % gs:
            fail(subj, "S", f"K={K} not divisible by groupsize {gs}")
            ok = False
        scale = cand.scale_f32(subj)
        if scale.shape != (N, 1):
            fail(subj, "S", f"scale shape {scale.shape} != ({N},1)")
            ok = False
        zrows = np.nonzero(scale[:, 0] <= np.float32(1e-30))[0]
        zbad = 0
        if len(zrows) and ok:
            codes = cand.codes_i8(subj).reshape(N, K)
            zbad = int((codes[zrows] != 0).sum())
            if zbad:
                fail(subj, "S0", f"{zbad} nonzero codes on {len(zrows)} zero-scale rows")
                ok = False
        n_side += 1
        if ok:
            details.append({"name": subj, "check": "S", "status": "PASS",
                            "groupsize": gs, "zero_scale_rows": int(len(zrows))})
    log(f"[S/S0] sidecar structure checked on {n_side} subjects, failures so far {len(failures)}")

    # ---- [P] -----------------------------------------------------------------
    pt_pass = pt_fail = pt_skip = 0
    for name in pt_keys:
        if only and not only.search(name):
            continue
        r = ref_lookup(name)
        ce = cand.ent(name)[1]
        if r is None:
            pt_skip += 1
            if not args.allow_missing_reference_keys:
                fail(name, "P", "no reference entry for pass-through key")
            continue
        re_ = r.ent(name)[1]
        if ce["dtype"] != re_["dtype"] or ce["shape"] != re_["shape"]:
            pt_fail += 1
            fail(name, "P", f"dtype/shape {ce['dtype']}{ce['shape']} vs "
                 f"ref {re_['dtype']}{re_['shape']}")
            continue
        hc, hr = cand.sha256(name), r.sha256(name)
        if hc == hr:
            pt_pass += 1
            details.append({"name": name, "check": "P", "status": "PASS", "sha256": hc})
        else:
            pt_fail += 1
            fail(name, "P", f"payload sha256 {hc[:16]}.. != ref {hr[:16]}..")
            details.append({"name": name, "check": "P", "status": "FAIL",
                            "sha256": hc, "ref_sha256": hr})
    log(f"[P] pass-through: {pt_pass} byte-identical, {pt_fail} mismatched, {pt_skip} skipped")

    # ---- [Q]/[L] -------------------------------------------------------------
    q_done = 0
    cos_i, cos_g = [], []
    qcos_all = []
    for subj, gs in sorted(quant_subs.items()):
        if only and not only.search(subj):
            continue
        q_done += 1
        if args.limit and q_done > args.limit:
            break
        N = int(cand.ent(subj)[1]["shape"][0])
        K = int(np.prod(cand.ent(subj)[1]["shape"][1:]))
        scale = cand.scale_f32(subj)
        codes_mm = cand.codes_i8(subj)
        r = ref_lookup(subj)
        if r is None:
            notes.append(f"{subj}: no reference (numeric/layout skipped)")
            details.append({"name": subj, "check": "Q", "status": "SKIP"})
            continue
        if r.ent(subj)[1]["shape"] != cand.ent(subj)[1]["shape"]:
            fail(subj, "Q", f"shape {cand.ent(subj)[1]['shape']} vs ref {r.ent(subj)[1]['shape']}")
            continue
        rctx = None
        if r.ent(subj)[1]["dtype"] == "I8":
            gs_r, _ = r.groupsize(subj)
            rctx = (gs_r, r.scale_f32(subj))
        elif r.ent(subj)[1]["dtype"] not in ("BF16", "F32", "F64"):
            notes.append(f"{subj}: reference dtype {r.ent(subj)[1]['dtype']} unsupported")
            details.append({"name": subj, "check": "Q", "status": "SKIP"})
            continue

        is_qkv = bool(QKV_PAT.search(subj)) and N % 3 == 0
        inner = N // 3

        def dequant_chunk(r0, r1):
            codes = codes_mm[r0 * K: r1 * K].reshape(r1 - r0, K)
            x = torch.from_numpy(codes.astype(np.float32) * scale[r0:r1])
            # rotate_stages_f32 already applies the NORMALIZED Hadamard
            # (h/sqrt(gs)); an extra /sqrt(gs) here broke relrms by 1-1/gs.
            return (rotate_stages_f32(x.reshape(r1 - r0, K // gs, gs)).numpy()
                    .reshape(r1 - r0, K))

        CH = max(1, min(N, (1 << 24) // max(K, 1)))
        acc_cr = acc_cc = acc_rr = 0.0
        acc_cr_g = acc_rr_g = 0.0
        Rg_full = None
        if is_qkv:
            # grouped row j -> HF interleaved row: per-128-row head blocks
            # (head = (j%inner)//128, kind = j//inner, sub = j%128):
            # src[j] = (head*3 + kind)*128 + sub  (== qkv_grouped_perm map)
            src = qkv_grouped_perm(np.arange(N, dtype=np.int64))
        for r0 in range(0, N, CH):
            r1 = min(N, r0 + CH)
            C = dequant_chunk(r0, r1)
            R = r.rows_f32(subj, r0, r1, rctx)
            acc_cr += float((C * R).sum())
            acc_cc += float((C * C).sum())
            acc_rr += float((R * R).sum())
            if is_qkv:
                Rg = r.rows_f32(subj, 0, N, rctx)[:, :]
                Rg = Rg[src[r0:r1]]
                acc_cr_g += float((C * Rg).sum())
                acc_rr_g += float((Rg * Rg).sum()) if r0 == 0 else acc_rr_g
                del Rg
            del C, R
        if is_qkv:
            acc_rr_g = acc_rr  # permutation preserves ||R||
        cos_f = acc_cr / max(1e-30, math.sqrt(acc_cc * acc_rr))
        relrms = math.sqrt(max(0.0, acc_cc + acc_rr - 2 * acc_cr)) / max(1e-30, math.sqrt(acc_rr))
        qcos_all.append(cos_f)
        rec = {"name": subj, "groupsize": gs, "cos_fileorder": round(cos_f, 6),
               "relrms": round(relrms, 6)}
        if is_qkv:
            cos_gr = acc_cr_g / max(1e-30, math.sqrt(acc_cc * acc_rr))
            cos_map = {"inter": cos_f, "grouped": cos_gr}
            cos_i.append(cos_f)
            cos_g.append(cos_gr)
            rec["cos_grouped_hypothesis"] = round(cos_gr, 6)
            if args.expect_layout == "any":
                ok = max(cos_map.values()) >= args.cos_min
                layout = max(cos_map, key=lambda k: cos_map[k])
            else:
                layout = args.expect_layout
                ok = cos_map[layout] >= args.cos_min
            if not ok:
                tag, diff = classify_permutation(subj, gs, r)
                rec["code_multiset_mismatch_frac"] = diff
                fail(subj, "L", f"inter={cos_map['inter']:.6f} grouped={cos_map['grouped']:.6f} "
                     f"expect={layout}", tag)
            details.append({**rec, "check": "L", "status": "PASS" if ok else "FAIL",
                            "match_layout": max(cos_map, key=lambda k: cos_map[k])})
        else:
            ok = cos_f >= args.cos_min and relrms <= args.relrms_max
            if not ok:
                tag, diff = classify_permutation(subj, gs, r)
                rec["code_multiset_mismatch_frac"] = diff
                fail(subj, "Q", f"cos={cos_f:.6f} relrms={relrms:.6f}", tag)
            details.append({**rec, "check": "Q", "status": "PASS" if ok else "FAIL"})
        if q_done % 50 == 0:
            log(f"[Q/L] {q_done}/{len(quant_subs)} quantized checked, failures {len(failures)}")

    summary = {
        "candidate": args.candidate,
        "references": args.reference,
        "n_quantized": len(quant_subs), "n_pass_through": len(pt_keys),
        "pass_through_pass": pt_pass, "pass_through_fail": pt_fail,
        "pass_through_skip": pt_skip,
        "failures": len(failures),
        "expect_layout": args.expect_layout,
        "verdict": "FAIL" if failures else "PASS",
    }
    if qcos_all:
        a = np.array(qcos_all)
        summary["quant_cos_fileorder"] = {"n": len(a), "min": float(a.min()),
                                          "p05": float(np.percentile(a, 5)),
                                          "median": float(np.median(a)),
                                          "below_cos_min": int((a < args.cos_min).sum())}
    if cos_i:
        ai, ag = np.array(cos_i), np.array(cos_g)
        summary["qkv_layout"] = {"n": len(ai),
                                 "inter_min": float(ai.min()), "inter_mean": float(ai.mean()),
                                 "grouped_min": float(ag.min()), "grouped_max": float(ag.max())}
    log("SUMMARY " + json.dumps(summary))
    if args.detail_out:
        with open(args.detail_out, "w") as f:
            json.dump({"summary": summary,
                       "failures": [list(x) for x in failures],
                       "details": details}, f, indent=1)
    for r_ in refs:
        r_.close()
    cand.close()
    if failures and not args.report_only:
        sys.exit(2)


if __name__ == "__main__":
    main()
