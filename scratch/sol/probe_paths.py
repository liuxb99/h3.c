import sys, glob; sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
from st_scan import open_st, fused_keys
import re

PATHS = [
    "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_2step_dwd_adapter.safetensors",
    "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_4step_dwd_adapter_int8_convrot.safetensors",
    "/Volumes/SSD2T/models/Pika h3 flex加速模型/models/diffusion_models/Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors",
]
for extra in [glob.glob("/Volumes/SSD2T/models/Pika h3 flex加速模型/models/**/*.safetensors", recursive=True),
              glob.glob("/Volumes/SSD2T/models/VPipe-comfy_h3/diffusion_models/*.safetensors"),
              glob.glob("/Volumes/SSD2T/models/VPipe-MiniMax-H3-turbo-lane/**/*.safetensors", recursive=True)]:
    PATHS += extra
seen = set()
for p in PATHS:
    if p in seen or not p.endswith(".safetensors") or not __import__("os").path.isfile(p):
        continue
    seen.add(p)
    st = open_st(p)
    import collections
    patsum = collections.Counter()
    for k, dt, shp in fused_keys(st):
        base = re.sub(r"\.\d+\.", ".N.", k)
        patsum[(base, dt, tuple(shp))] += 1
    scales = sum(1 for k in st["hdr"] if k.endswith("weight_scale"))
    cqs = sum(1 for k in st["hdr"] if k.endswith("comfy_quant"))
    print(f"== {p.split('/')[-1]} keys={len(st['hdr'])} scales={scales} cq={cqs}")
    for (base, dt, shp), n in sorted(patsum.items()):
        print(f"   x{n:<3} {base}  {dt} {shp}")
    st["f"].close()
