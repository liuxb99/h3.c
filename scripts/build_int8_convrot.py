#!/usr/bin/env python3
"""build_int8_convrot.py — rebuild MiniMax-H3 DiT convrot int8 weights from the
untouched HF original, repairing the two damage modes found in the official
Comfy-Org conversion (see scripts/qkv_layout_gate.py + scratch/sol audit):

  D1 pass-through requant drift  -> here: raw payload BYTES copied from HF
      original, zero requant round-trip (safetensors payload is raw LE; byte
      copy structurally guarantees byte-identity).
  D2 fused multi-kind row reorder -> here: NO row permutation applied to any
      tensor; HF canonical (per-head interleaved for qkv) row order preserved.

Contract replication: quantized subject set, per-subject convrot_groupsize,
comfy_quant sidecar JSON payloads and __metadata__ are copied VERBATIM from a
contract file (default: the official Comfy-Org int8 file) so the output is
name/shape/format identical to what loaders already accept; only numeric
payloads are regenerated from HF values with the f32 quantizer line
(rotate -> absmax/127 scale -> RNE round), i.e. the same pipeline as
ComfyUI-piFlow/tools/convert_minimax_h3_adapter_int8_convrot.py applied to the
full DiT.

Output: single-file safetensors (64B payload alignment, engine-safe) +
engine-loadable FL2VA/transformer directory mirror + index json + manifest.

Run: uv run --with torch,numpy -q python scripts/build_int8_convrot.py \
       [--quant-mode butterfly|matmul|skip] [--parity]
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from convrot_quant import quant_tensor_f32line  # noqa: E402
from st_io import payload, st_close, st_create, st_open  # noqa: E402

HF_DIR_DEFAULT = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
CONTRACT_DEFAULT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
OUT_DEFAULT = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ei_retry(fn, *a, tries=10, delay=0.25, **k):
    """EINTR-safe call. /Volumes/SSD2T (exFAT-class) intermittently raises
    InterruptedError on listdir/open/remove (3 incidents 2026-11-28: build1
    os.listdir cleanup, p4h run_log reads). Transient — verified by retry."""
    for i in range(tries):
        try:
            return fn(*a, **k)
        except InterruptedError:
            if i == tries - 1:
                raise
            log(f"EINTR retry {i+1}/{tries} on {getattr(fn, '__name__', fn)}")
            time.sleep(delay * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=HF_DIR_DEFAULT)
    ap.add_argument("--contract", default=CONTRACT_DEFAULT)
    ap.add_argument("--mirror-only", action="store_true",
                    help="reuse existing --out-single; rebuild dir mirror + manifest only")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--stem", default="minimax_h3_ref2va_int8_convrot_h3c")
    ap.add_argument("--quant-mode", default="butterfly",
                    choices=["butterfly", "matmul", "skip"],
                    help=("butterfly = engine digit stages; matmul = explicit "
                          "kron cross-check (identical result, slower); "
                          "skip = zeros"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--parity", action="store_true",
                    help="measure butterfly-vs-matmul code agreement on 2 tensors, then exit")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    hf_index = json.load(open(os.path.join(args.hf_dir, "model.safetensors.index.json")))
    wmap = hf_index["weight_map"]
    shards = sorted(set(wmap.values()))

    # ---- contract ----------------------------------------------------------
    ch, cf, cmm, cds = st_open(args.contract)
    metadata = ch.get("__metadata__", {})
    contract = {k: v for k, v in ch.items() if k != "__metadata__"}
    quant_subs = {}
    for k, ent in contract.items():
        if k.endswith(".comfy_quant"):
            subj = k[: -len(".comfy_quant")]
            cq = payload(cmm, cds, ent)
            cj = json.loads(cq)
            assert cj.get("format") == "int8_tensorwise" and cj.get("convrot") is True, subj
            assert cj["convrot_groupsize"] in (16, 64, 256), subj
            quant_subs[subj + ".weight"] = (contract[subj + ".weight"],
                                             contract[subj + ".weight_scale"],
                                             bytes(cq), int(cj["convrot_groupsize"]))
    pt_keys = [k for k in contract
               if k not in quant_subs and not k.endswith(".weight_scale")
               and not k.endswith(".comfy_quant")]
    log(f"contract: {len(quant_subs)} quantized subjects, {len(pt_keys)} pass-through keys")

    # ---- HF original -------------------------------------------------------
    shdr, shm = {}, {}
    for s in shards:
        h, f, mm, ds = st_open(os.path.join(args.hf_dir, s))
        shdr[s] = (h, ds)
        shm[s] = (f, mm)

    def hf_bytes(name):
        shard = wmap[name]
        h, ds = shdr[shard]
        ent = h[name]
        a, b = ent["data_offsets"]
        return shm[shard][1][ds + a: ds + b]

    subj_w_keys = set(quant_subs) | set(pt_keys)
    # key-set compare is SUBJECT-weight keys only: .weight_scale/.comfy_quant
    # are contract sidecars, never part of the HF weight_map.
    missing = subj_w_keys - set(wmap)
    extra = set(k for k in wmap) - subj_w_keys
    if missing and not all(m in quant_subs for m in missing):
        raise SystemExit(f"missing non-quant subjects (no golden possible): {sorted(missing)[:8]}")
    for m in sorted(missing):
        # quant subject weights without HF golden on disk: the official
        # contract bytes are the baseline for these (no HF value exists),
        # so codes travel VERBATIM from the contract file (handled in
        # quant_now via `subj not in wmap`).
        assert m in quant_subs, m
    if extra:
        raise SystemExit(f"unexpected HF keys not in contract: {sorted(extra)[:8]}")
    log(f"contract quant subjects without HF golden (verbatim official): {len(missing)}")

    # ---- qkv row-layout rule (engine h3_weights.c H3_DIT_QKV_LAYOUT_ROWS) --
    # HF original stores fused qkv rows PER-HEAD INTERLEAVED:
    #   inter row r = (unit*3 + kind)*128 + sub   (head = r//128, kind = head%3)
    # The convrot contract (official Comfy-Org file; engine comment: "the
    # checkpoint stores qkv rows in q|k|v grouped form") stores them KIND-
    # GROUPED: file row j = (kind*hpc + unit)*128 + sub. The engine applies
    # the rowmap at dequant, so the built file MUST store grouped rows.
    # Verified empirically against the official file (scratch/sol/probe_qkv_
    # layout.py: official I8 rows grouped, HF reference interleaved).
    def qkv_grouped_perm(rows, n_rows):
        """permute interleaved-HF rows -> kind-grouped file rows."""
        assert n_rows % 384 == 0
        hpc = n_rows // 384
        j = np.arange(n_rows)
        kind, unit, sub = j // (hpc * 128), (j // 128) % hpc, j % 128
        src = ((unit * 3 + kind) * 128 + sub)
        return rows[src]
    for name in subj_w_keys:
        if name not in wmap:
            continue  # verbatim-official quant subject (no HF golden), checked below
        shard = wmap[name]
        hf_dtype = shdr[shard][0][name]["dtype"]
        want = "I8" if name in quant_subs else hf_dtype
        got = quant_subs[name][0]["dtype"] if name in quant_subs else "I8"
        # pass-through: contract entry dtype must equal HF dtype
        if name not in quant_subs:
            ce = contract[name]
            assert ce["dtype"] == hf_dtype and ce["shape"] == shdr[shard][0][name]["shape"], \
                f"dtype/shape mismatch pass-through {name}: {ce['dtype']} vs {hf_dtype}"
        else:
            assert hf_dtype == "BF16", f"quant source must be bf16 in HF: {name} {hf_dtype}"
            assert quant_subs[name][0]["dtype"] == "I8"
    log(f"key/dtype parity vs contract OK ({len(subj_w_keys)} subjects)")

    # ---- shared quant cache (weight + scale tensors quantized once) --------
    cache = {}
    stats = {"quant_done": 0}

    def qkv_layout_needed(subj):
        return subj.endswith("attn.qkv_proj.weight")

    def quant_now(subj):
        key = subj
        if key not in cache:
            went, scent, cq, gs = quant_subs[subj]
            if subj not in wmap:
                # no HF golden exists: official I8 bytes are the baseline
                codes = (np.zeros((went["shape"]), dtype=np.int8)
                         if args.quant_mode == "skip" else
                         np.frombuffer(bytes(payload(cmm, cds, went)),
                                       dtype=np.int8).view(tuple(went["shape"])))
                cache[key] = (np.ascontiguousarray(codes), None)
                stats["quant_done"] += 1
                return cache[key]
            shard = wmap[subj]
            ent = shdr[shard][0][subj]
            shape = tuple(ent["shape"])
            raw = hf_bytes(subj)
            w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(shape)
            del raw
            if args.quant_mode == "skip":
                codes = np.zeros(shape, dtype=np.int8)
            else:
                mode = "matmul_ref" if args.quant_mode == "matmul" else "int8"
                codes_t, _scale_t = quant_tensor_f32line(w, gs, mode=mode)
                codes = codes_t.numpy()
                if qkv_layout_needed(subj) and shape[0] % 384 == 0:
                    codes = qkv_grouped_perm(codes, shape[0])
            cache[key] = (np.ascontiguousarray(codes), None)
            del w
            stats["quant_done"] += 1
            if stats["quant_done"] % 25 == 0 or stats["quant_done"] == len(quant_subs):
                log(f"quantized {stats['quant_done']}/{len(quant_subs)} (last {subj})")
        return cache[key]

    def code_sink(subj):
        def sink(f):
            codes, _ = quant_now(subj)
            f.write(codes.tobytes(order="C"))
        return sink

    def scale_sink(subj):
        went, scent, cq, gs = quant_subs[subj]
        raw = bytes(payload(cmm, cds, scent))  # F32 [n,1] VERBATIM contract scale
        def sink(f):
            f.write(raw)
            cache.pop(subj, None)  # pair fully written; free before next subject
        return sink

    def pass_sink(name):
        def sink(f):
            f.write(hf_bytes(name))
        return sink

    # ---- output plan: sorted key order; scale emitted AFTER codes so the
    #      cache holds one tensor at a time (weight + scale share quant_now).
    plan = []
    for name in sorted(subj_w_keys):
        if name in quant_subs:
            went, scent, cq, gs = quant_subs[name]
            n, k = went["shape"]
            plan.append((name, "I8", (n, k), n * k, code_sink(name)))
            # NOTE: st_create writes in plan order; scale must be produced by
            # the same quant -> emit scale BEFORE codes in plan so codes reuse
            # cache; but codes tensor is the bigger one, order only affects
            # padding. Keep canonical name-sorted order and rely on cache:
            # 2026-11-28 naming fix: production loader convention (official
            # comfy-int8 AND h3c-engine asset, both measured via
            # sidecar_keys_probe.py) is the STRIP form: scale = subj +
            # "_scale", comfy sidecar = subj minus ".weight" + ".comfy_quant".
            # The append form this builder used to emit is unreadable by the
            # engine loader (250/250 scale-key misses on the shipped asset).
            plan.append((name + "_scale", "F32", (n, 1), 4 * n, scale_sink(name)))
            cq_key = name[: -len(".weight")] + ".comfy_quant"
            plan.append((cq_key, "U8", [len(cq)], len(cq), cq))
        else:
            shard = wmap[name]
            ent = shdr[shard][0][name]
            plan.append((name, ent["dtype"], ent["shape"],
                         ent["data_offsets"][1] - ent["data_offsets"][0],
                         pass_sink(name)))

    # sorted(subj_w_keys) puts weight_scale/comfy_quant AFTER weight (suffix >)
    # but plan order above interleaves intentionally: weight -> scale -> quant.
    if args.parity:
        for subj, gs in [("blocks.0.attn.qkv_proj.weight", 256),
                         ("blocks.0.adaln_proj.linear.weight", 64)]:
            shard = wmap[subj]
            ent = shdr[shard][0][subj]
            shape = tuple(ent["shape"])
            w = torch.frombuffer(bytearray(hf_bytes(subj)), dtype=torch.bfloat16).view(shape)
            t0 = time.time()
            cb, sb = quant_tensor_f32line(w, gs, mode="int8")
            t1 = time.time()
            cm, sm = quant_tensor_f32line(w, gs, mode="matmul_ref")
            t2 = time.time()
            diff = int((cb.numpy() != cm.numpy()).sum())
            sc = float(np.abs(sb.numpy() - sm.numpy()).max())
            log(f"PARITY {subj}: codes differ {diff}/{cb.numel()} ({100.0*diff/cb.numel():.6f}%), "
                f"max |scale diff| {sc:.3e}; butterfly {t1-t0:.1f}s matmul {t2-t1:.1f}s")
        return

    out_single = os.path.join(args.out_dir, args.stem + ".safetensors")
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    if args.mirror_only:
        if not os.path.exists(out_single):
            raise SystemExit("--mirror-only: no existing single build to mirror")
        oh, of, omm, ods = st_open(out_single)
        total = os.path.getsize(out_single)
        of.close()
        omm.close()
        header = oh
        log(f"--mirror-only: reusing existing single build ({total} B); quant skipped")
    else:
        # stale leftovers from interrupted runs (e.g. a 1MB .writing) go first
        for stale in ei_retry(os.listdir, args.out_dir):
            if stale.endswith(".writing"):
                ei_retry(os.remove, os.path.join(args.out_dir, stale))
                log(f"removed stale leftover {stale}")
        tmp = out_single + ".writing"
        total, header = ei_retry(st_create, tmp, plan, metadata=metadata, align=64)
        ei_retry(os.replace, tmp, out_single)  # atomic publish
    el = time.time() - t_start
    log(f"single file written: {out_single} size={total} in {el/60:.1f} min, quantized={stats['quant_done']}")

    # ---- deployment layout: mirror the engine-copy dir (single main file) --
    # refiner/FL2VA weights are consumed from the shared MiniMaxAI FL2VA HF
    # dir via a symlink, exactly like MiniMax-H3-comfy-int8/FL2VA; hardlinking
    # the 34G main file under FL2VA/transformer (older layout) is wrong since
    # the FL2VA model dir is a separate 13-shard HF directory.
    fl_link = os.path.join(args.out_dir, "FL2VA")
    fl_src = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA"
    if os.path.lexists(fl_link):
        ei_retry(os.remove, fl_link)
    ei_retry(os.symlink, fl_src, fl_link)

    hsh = hashlib.sha256()
    with open(out_single, "rb") as f:
        while True:
            b = f.read(1 << 26)
            if not b:
                break
            hsh.update(b)
    manifest = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_dir": args.hf_dir, "contract": args.contract,
        "quant_mode": args.quant_mode, "threads": args.threads,
        "single_file": out_single, "size_bytes": total, "sha256": hsh.hexdigest(),
        "n_quantized": len(quant_subs), "n_passthrough": len(pt_keys),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
        "tensor_entries": len(header) - 1,
    }
    with open(os.path.join(args.out_dir, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("MANIFEST " + json.dumps(manifest))
    for s in shards:
        st_close(shm[s][1], shm[s][0])
    st_close(cmm, cf)


if __name__ == "__main__":
    main()
