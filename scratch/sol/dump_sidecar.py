import json, struct, sys
p="/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
f=open(p,"rb")
n=struct.unpack("<Q", f.read(8))[0]
hdr=json.loads(f.read(n))
for k in ["blocks.0.attn.qkv_proj.comfy_quant","blocks.0.attn.out_proj.comfy_quant","blocks.0.mlp.fc1.comfy_quant","blocks.0.attn.out_proj.weight","blocks.0.attn.out_proj.weight_scale"]:
    v=hdr.get(k)
    print("KEY", k, "->", v if v is None or k.endswith("comfy_quant") is False else "")
    if k.endswith("comfy_quant") and v:
        off=v["data_offsets"][0]
        f.seek(8+n+off)
        print("   JSON:", f.read(v["data_offsets"][1]-off).decode())
    elif v: print("   meta:", v)
