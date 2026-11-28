#!/usr/bin/env python3
"""Standalone final-head semantic check (no block loop): engine b49 block_out taps feed
the engine-exact final AdaLN + velocity linear; compare to engine taps."""
import glob, json, os, struct, mmap
import numpy as np, torch
D = "/Users/cyliu99/.hermes/cache/scratch/fulldit_dump"
CKPT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
HIDDEN, VP, VSTART, VROWS = 5376, 96, 135, 48
_BF16, _F32 = torch.bfloat16, torch.float32
def tap(n):
    h = sorted(glob.glob(os.path.join(D, n + "_*.raw")))[0]
    return torch.from_numpy(np.frombuffer(open(h, "rb").read(), dtype="<u2").copy()).view(_BF16)
_f = open(CKPT, "rb"); _ln = struct.unpack("<Q", _f.read(8))[0]
hdr = json.loads(_f.read(_ln)); _mm = mmap.mmap(_f.fileno(), 0, access=mmap.ACCESS_READ); _DS = 8 + _ln
def ckpt_tensor(name):
    i = hdr[name]; a, b = i["data_offsets"]
    buf = bytearray(_mm[_DS+a:_DS+b])
    dt = {"BF16": _BF16, "F32": _F32}[i["dtype"]]
    return torch.frombuffer(buf, dtype=dt).reshape(i["shape"])
gamma = ckpt_tensor("final_layer.norm.weight").to(_BF16)
bias = ckpt_tensor("final_layer.video_out.bias").to(_BF16)
vw32 = ckpt_tensor("final_layer.video_out.weight").reshape(VP, HIDDEN)
def rel(a, b):
    a, b = a.reshape(-1).float().double(), b.reshape(-1).float().double()
    return float((a-b).norm()/b.norm())
for step in (0, 1):
    hid = tap(f"b49_s{step}_block_out").view(183, HIDDEN)[VSTART:VSTART+VROWS]
    x = hid.float()
    fn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma.float()
    fm = tap(f"s{step}_final_mod_bf16").view(-1, 2*HIDDEN)
    vmap = np.frombuffer(open(sorted(glob.glob(os.path.join(D, f"s{step}_final_video_map_u32_*.raw")))[0], "rb").read(), dtype="<u4")
    rows = torch.from_numpy(vmap.astype("int64"))
    adaln = (fn * (1.0 + fm[rows, HIDDEN:].float()) + fm[rows, :HIDDEN].float()).to(_BF16)
    eng = tap(f"s{step}_final_video_norm_bf16")
    print(f"step{step} final_video_norm rel={rel(adaln, eng):.3e} cos_ok shapes {tuple(adaln.shape)} vs {tuple(eng.shape)}")
    n = adaln.float()
    cands = {"w_f32": (n @ vw32.t() + bias.float()).to(_BF16),
             "w_bf16": (n @ vw32.to(_BF16).float().t() + bias.float()).to(_BF16)}
    for k, v in cands.items():
        print(f"step{step} velocity[{k}] rel={rel(v, tap(f's{step}_velocity_bf16')):.3e}")
