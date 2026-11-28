# Staged row-order measurement driver. Usage: measure.py <stage> -> appends JSONL
import sys, json, glob
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import numpy as np
from measure_lib import (open_st, hf_tensor, run_family, measure, hf_u16_ms,
                         asset_u16_ms, deq_bf16, dequant_unrot, cos_sim, _find_shard)
from st_scan import grouped_from_interleaved, interleave_from_grouped

OUT = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results_%s.jsonl"
STAGE = sys.argv[1]

def emit(out):
    with open(OUT % STAGE, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"stage {STAGE}: wrote {len(out)} records")

out = []
if STAGE == "1_hf_official":
    IDX = [0, 25, 49]
    # HF baseline (self-signature; interleaved established)
    names = [f"blocks.{i}.attn.qkv_proj.weight" for i in IDX] + \
            ["token_refiner.blocks.0.attn.qkv_proj.weight",
             "token_refiner.blocks.1.attn.qkv_proj.weight"] + \
            [f"blocks.{i}.mlp.fc1.weight" for i in IDX] + \
            ["token_refiner.blocks.0.mlp.fc1.weight", "token_refiner.blocks.1.mlp.fc1.weight"]
    import measure_lib as ml
    for n in names:
        ref = hf_tensor(n)
        kind = "qkv3" if "qkv" in n else "fc1"
        st = open_st(ml._find_shard(n))
        ml.measure(st, n, ref=ref, kind=kind, quant="bf16", tag="ORIG_HFL2VA", out=out)
        st["f"].close(); del ref
    # official ref2va convrot
    st = open_st("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
    run_family(st, [f"blocks.{i}.attn.qkv_proj.weight" for i in IDX], "OFFICIAL_CONVROT_REF2VA", "convrot", "qkv3", out)
    run_family(st, [f"blocks.{i}.mlp.fc1.weight" for i in IDX], "OFFICIAL_CONVROT_REF2VA", "convrot", "fc1", out)
    run_family(st, ["token_refiner.blocks.0.attn.qkv_proj.weight",
                    "token_refiner.blocks.1.attn.qkv_proj.weight",
                    "token_refiner.blocks.0.mlp.fc1.weight",
                    "token_refiner.blocks.1.mlp.fc1.weight"],
               "OFFICIAL_CONVROT_REF2VA", "bf16", "qkv3", out, multiset=True)
    # refiner fc1 needs kind fc1
    st2 = open_st("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
    run_family(st2, ["token_refiner.blocks.0.mlp.fc1.weight", "token_refiner.blocks.1.mlp.fc1.weight"],
               "OFFICIAL_CONVROT_REF2VA", "bf16", "fc1", out, multiset=True)
    st2["f"].close()
    emit(out)

elif STAGE == "1b_comfy_bf16":
    P = "/Volumes/SSD2T/models/Comfy-Org-MiniMax-H3-bf16/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    st = open_st(P)
    names = sorted(k for k in st["hdr"] if ("attn.qkv_proj.weight" in k or "mlp.fc1.weight" in k)
                   and not k.endswith("weight_scale") and not k.endswith("comfy_quant"))
    main_q = [k for k in names if "qkv" in k and k.startswith("blocks.")]
    main_f = [k for k in names if "fc1" in k and k.startswith("blocks.")]
    ref_q = [k for k in names if "qkv" in k and k.startswith("token_refiner")]
    ref_f = [k for k in names if "fc1" in k and k.startswith("token_refiner")]
    run_family(st, main_q, "COMFY_ORG_BF16_FL2VA", "bf16", "qkv3", out, multiset=True)
    run_family(st, main_f, "COMFY_ORG_BF16_FL2VA", "bf16", "fc1", out, multiset=True)
    run_family(st, ref_q, "COMFY_ORG_BF16_FL2VA", "bf16", "qkv3", out, multiset=True)
    run_family(st, ref_f, "COMFY_ORG_BF16_FL2VA", "bf16", "fc1", out, multiset=True)
    st["f"].close()
    emit(out)
