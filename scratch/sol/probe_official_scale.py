#!/usr/bin/env python3
"""probe_official_scale.py — settle sidecar SCALE semantics against OFFICIAL.

Candidates for the official stored per-row scale (computed from the HF
original with the settled conference-h4 line):
  A upstream_line : scale = clamp(absmax(rot_bf16)/127, 1e-30)
                    (rot = h4-kron /sqrt(G) rotation in bf16)
  B folded        : scale = absmax(HF_row_unrotated) / (127*sqrt(G))
Outputs per-subject byte/rel match of official scale vs each candidate +
code exact/within1. Writes scratch/sol/probe_official_scale.json.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
sys.path.insert(0, os.path.join(HERE, "scripts"))
import convrot_quant as cq  # noqa: E402
from st_io import payload, st_close, st_open  # noqa: E402

OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
OUT = os.path.join(HERE, "scratch/sol/probe_official_scale.json")

SAMPLE = [
    "blocks.0.attn.qkv_proj.weight",      # gs 256, grouped-order question
    "blocks.0.adaln_proj.linear.weight",  # gs 64
    "blocks.10.mlp.fc2.weight",           # gs 256
]


def grouped_perm(n_rows, head_dim=128, n_heads=56):
    """Map grouped FILE rows -> interleaved HF rows (file_row j -> hf row)."""
    assert n_rows % (3 * head_dim) == 0
    hpc = n_rows // (3 * head_dim)
    j = np.arange(n_rows)
    kind, unit, sub = j // (hpc * head_dim), (j // head_dim) % hpc, j % head_dim
    return (unit * 3 + kind) * head_dim + sub


def main():
    hsize = os.path.getsize(OFFICIAL)
    head = hashlib.sha256(open(OFFICIAL, "rb").read(1 << 20)).hexdigest()[:16]
    print(f"OFFICIAL = {OFFICIAL}")
    print(f"size={hsize} mtime={time.ctime(os.path.getmtime(OFFICIAL))} headsha={head}")

    torch.set_num_threads(8)
    wmap = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))["weight_map"]
    shm, shdr = {}, {}

    def hf_tensor(name):
        shard = wmap[name]
        if shard not in shm:
            h, f, mm, ds = st_open(os.path.join(HF_DIR, shard))
            shdr[shard], shm[shard] = (h, ds), (f, mm)
        h, ds = shdr[shard]
        ent = h[name]
        a, b = ent["data_offsets"]
        raw = bytes(shm[shard][1][ds + a: ds + b])
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(tuple(ent["shape"]))

    oh, of, omm, ods = st_open(OFFICIAL)
    result = {"official": OFFICIAL, "official_size": hsize,
              "official_head_sha256_1MB": head, "subjects": {}}

    for name in SAMPLE:
        stem = name[: -len(".weight")]
        gs = int(json.loads(payload(omm, ods, oh[stem + ".comfy_quant"]))["convrot_groupsize"])
        s_ent = oh[stem + ".weight_scale"]
        off_scale = torch.frombuffer(bytearray(payload(omm, ods, s_ent)),
                                     dtype=torch.float32).view(-1).numpy()
        w = hf_tensor(name)
        n, k = w.shape
        codes, scale = cq.quant_tensor_f32line(w, gs, mode="int8")
        scale = np.asarray(scale.numpy())
        folded = (w.abs().amax(dim=1).float().numpy()) / (127.0 * math.sqrt(gs))
        assert off_scale.shape[0] == n

        cands = {"A_direct": scale, "B_folded_direct": folded,
                 "C_raw_unnorm": scale * math.sqrt(gs)}
        if stem.endswith("attn.qkv_proj"):
            perm = grouped_perm(n)
            cands["A_grouped"] = scale[perm]
            cands["B_folded_grouped"] = folded[perm]
            cands["C_raw_grouped"] = scale[perm] * math.sqrt(gs)
        for cname, cand in list(cands.items()):
            cand = np.asarray(cand, dtype=np.float32)
            rel = np.abs(cand - off_scale) / np.maximum(off_scale, 1e-38)
            cands[cname] = {"byte_eq_frac": float((cand == off_scale).mean()),
                            "max_rel": float(rel.max()),
                            "mean_rel": float(rel.mean())}
        oc = torch.frombuffer(bytearray(payload(omm, ods, oh[name])),
                              dtype=torch.int8).view(n, k).numpy()
        myc = codes.numpy()
        if stem.endswith("attn.qkv_proj"):
            myc = myc[grouped_perm(n), :]
        dev = np.abs(myc.astype(np.int32) - oc.astype(np.int32))
        result["subjects"][name] = {
            "gs": gs, "scale_shape_official": list(s_ent["shape"]),
            "scale": cands,
            "codes_exact_frac": float((dev == 0).mean()),
            "codes_within1_frac": float((dev <= 1).mean()),
        }
        print(name, json.dumps(result["subjects"][name], indent=1))
        del w, codes, oc, myc

    ev = os.path.join(HERE, "docs/evidence/2026-11-25_convrot三方驗收報告.json")
    if os.path.exists(ev):
        t = json.load(open(ev))["tensors"]["blocks.0.attn.qkv_proj.weight"]
        result["engine_contract_evidence"] = {
            "source": ev, "upstream_bit_equal": t["upstream_bit_equal"],
            "cpu_vs_ref_exact": t["cpu_vs_ref_exact"],
            "gpu_vs_ref_max_dev_quant_steps": t["gpu_vs_ref_max_dev_quant_steps"],
        }
    st_close(omm, of)
    for f, mm in shm.values():
        mm.close()
        f.close()
    json.dump(result, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
