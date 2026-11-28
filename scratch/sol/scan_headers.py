# Step 1: enumerate fused-projection keys + dtypes + sidecars per asset
import glob
import sys
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
from st_scan import open_st, fused_keys

FILES = [
    ("ORIG_HFL2VA", "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/transformer/model-*.safetensors"),
    ("COMFY_CONVROT_REF2VA", "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"),
    ("COMFY_PRUNED_FL2VA", "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    ("COMFY_PRUNED_REF2VA", "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ("COMFY_FUSED_LORA", "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_fused_flex_lora_v1.safetensors"),
    ("COMFY_REQUANT_RT", "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_requant_roundtrip.safetensors"),
    ("ADAPTER", "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors"),
    ("TE_NVFP4", "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("TE_ORIG", "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/text_encoder/*.safetensors"),
    ("SULPHUR3", "/Volumes/SSD2T/models/SulphurAI-Sulphur-3/**/*.safetensors"),
    ("VPIPE8", "/Volumes/SSD2T/models/VPipe-FL2VA-8bit/diffusion_models/*.safetensors"),
]

import re
for tag, pat in FILES:
    paths = sorted(glob.glob(pat, recursive=True))
    if not paths:
        print(f"### {tag}: NO FILES ({pat})")
        continue
    print(f"### {tag}: {len(paths)} file(s)")
    for p in paths[:20]:
        st = open_st(p)
        fk = fused_keys(st)
        # aggregate key-pattern summary to keep output small
        import collections
        patsum = collections.Counter()
        for k, dt, shp in fk:
            base = re.sub(r"\.\d+\.", ".N.", k)
            patsum[(base, dt, tuple(shp))] += 1
        scales = [k for k in st["hdr"] if k.endswith("weight_scale")]
        cqs = [k for k in st["hdr"] if k.endswith("comfy_quant")]
        print(f"  {p.split('/')[-1]}: {len(st['hdr'])} keys, "
              f"{len(scales)} weight_scale, {len(cqs)} comfy_quant")
        for (base, dt, shp), n in sorted(patsum.items()):
            print(f"    x{n:<3} {base}  {dt} {shp}")
        st["f"].close()
    print()
