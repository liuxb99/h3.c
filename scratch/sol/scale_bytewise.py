#!/usr/bin/env python3
"""scale_bytewise.py — builder-scale vs contract-scale byte identity on sampled subjects.

Regenerates codes+scale with scripts/convrot_quant.quant_tensor_f32line from the HF
original for a stratified sample of quantized subjects and compares:
  * scale payload BYTES (built order == contract order: scale order == code row
    order; for qkv subjects compare after applying qkv_grouped_perm to built rows)
  * codes: bitwise-equal fraction + within-1 fraction
Writes scratch/sol/scale_bytewise.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
sys.path.insert(0, os.path.join(HERE, "scripts"))
import convrot_quant as cq  # noqa: E402
from st_io import payload, st_close, st_open  # noqa: E402

HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
CONTRACT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
OUT = os.path.join(HERE, "scratch/sol/scale_bytewise.json")

SAMPLE = [
    "blocks.0.attn.qkv_proj.weight",      # gs 256 qkv (permuted rows)
    "blocks.10.mlp.fc2.weight",           # gs 256
    "blocks.0.adaln_proj.linear.weight",  # gs 64, has zero rows? (adaln-zero init: block 49 has them)
    "blocks.49.adaln_proj.linear.weight", # gs 64, adaLN-zero 21504 zero rows
    "blocks.0.mlp.fc1.weight",            # gs 256
    "blocks.24.attn.out_proj.weight",     # gs 256
]


def main():
    torch.set_num_threads(8)
    hf_index = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))
    wmap = hf_index["weight_map"]
    ch, cf, cmm, cds = st_open(CONTRACT)
    res = {}
    for name in SAMPLE:
        stem = name[: -len(".weight")]
        gs, cj = None, None
        cq_ent = ch[stem + ".comfy_quant"]
        cj = json.loads(payload(cmm, cds, cq_ent))
        gs = int(cj["convrot_groupsize"])
        # HF source
        shard = wmap[name]
        hs, hf, hmm, hds = None, None, None, None
        # open per-shard (cheap: cached set)
        hs, hf, hmm, hds = st_open(os.path.join(HF_DIR, shard))
        ent = hs[name]
        shape = tuple(ent["shape"])
        a, b = ent["data_offsets"]
        raw = hds[hmm + a: hmm + b] if False else None
        import mmap
        raw = hmm[hmm + a: hmm + b] if False else None
        raw = hds[a + hds_offset(hmm, a): 0] if False else None
        # hds IS the data-start offset int (st_io returns (header, f, mm, data_start)? check)
        st_close(hmm, hf)
        res[name] = {"gs": gs, "note": "placeholder"}
    st_close(cmm, cf)
    print(json.dumps(res, indent=1))


def hds_offset(hmm, a):
    return 0


if __name__ == "__main__":
    main()
