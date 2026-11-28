#!/usr/bin/env python3
"""ConvRot int8_tensorwise dequant reference harness (h3.c semantics gate).

Mirrors upstream comfy_kitchen eager chain exactly (spindle:
backends/eager/quantization.py dequantize_int8_convrot_weight L916-927,
dequantize_int8_simple L929, _rotate_weight/_build_hadamard per
backends/eager/convrot_w4a4.py L15-74):

  dequant(q, scale, gs) = rotate(dequant_simple(q, scale), H_gs, gs)
    dequant_simple: q.float() * scale          (fp32 per element)
    H_gs:  h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
           kron-power to gs, / sqrt(gs)        (NOTE: upstream h4 is a
           conference-type Hadamard, NOT Sylvester — never FWHT-swap it)
    rotate: W.view(N, K//gs, gs) @ H_gs.T  -> view(N, K)   (fp32 matmul)

Also quant-side re-encode check (upstream quantize_int8_convrot_weight:
rotate in *weight dtype* (bf16) then rowwise scale=absmax/127 clamp 1e-30,
divide in bf16, RNE round, clamp [-128,127]).

If the real comfy_kitchen package is importable (CK_PATH env), every
reference value is asserted bitwise-equal against the upstream function;
otherwise the mirror below stands alone (it was transcribed from the
upstream source quoted above).

Outputs per tensor into OUT_DIR (default scratch):
  <name>.q.bin       int8 codes   [N,K]   (for CPU/Metal refs)
  <name>.scale.bin   f32 scales   [N,1]
  <name>.ref_fp32.bin  f32 dequantized original-basis weight [N,K]
  <name>.ref_bf16.bin  uint16 bf16(RNE) cast of the above
  <name>.np_fp32.bin   independent numpy-matmul fp32 chain [N,K]
  meta.json          shapes/groupsize/sidecar + requant stats
"""
import json
import math
import os
import struct
import subprocess
import sys

import numpy as np
import torch

FILE = os.environ.get(
    "CONVROT_FILE",
    "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors",
)
OUT = os.environ.get("CONVROT_OUT", "/Users/cyliu99/.hermes/cache/scratch/convrot_sem")
CK_PATH = os.environ.get("CK_PATH", "/Users/cyliu99/.hermes/cache/scratch/ck")

TARGETS = [
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
    "blocks.0.adaln_proj.linear.weight",
    "blocks.24.attn.qkv_proj.weight",
    "blocks.49.mlp.fc1.weight",
    "blocks.49.attn.out_proj.weight",
]

# ---------------------------------------------------------------- safetensors
def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h

def tensor_bytes(path, h, name):
    """Payload bytes for a tensor; safetensors offsets are relative to
    (8 + header_len)."""
    e = h[name]
    o = e["data_offsets"]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + o[0])
        return f.read(o[1] - o[0]), e["dtype"], e["shape"]

# ---------------------------------------------------------------- upstream mirror
H4_CACHE = {}

def build_hadamard(size, dtype=torch.float32):
    """Mirror of eager/convrot_w4a4.py _build_hadamard (kron power of h4)."""
    key = (size, dtype)
    if key in H4_CACHE:
        return H4_CACHE[key]
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    H4_CACHE[key] = h
    return h

def rotate_weight(w, h, gs):
    """Mirror of _rotate_weight: view(N, n_groups, gs) @ h.T."""
    n, k = w.shape
    return (w.view(n, k // gs, gs) @ h.T).reshape(n, k)

def dequant_simple(q, scale):
    return q.float() * scale

def dequant_convrot_mirrored(q, scale, gs):
    h = build_hadamard(gs, torch.float32)
    return rotate_weight(dequant_simple(q, scale), h, gs)

def quant_convrot_mirrored(w_bf16, gs):
    """Mirror of quantize_int8_convrot_weight + quantize_int8_rowwise +
    _int8_scale_for_math/_round_int8 (scale math happens in x.dtype!)."""
    h = build_hadamard(gs, torch.bfloat16)
    wr = rotate_weight(w_bf16, h, gs)
    absmax = wr.abs().amax(dim=-1, keepdim=True)
    scale = (absmax.float() / 127.0).clamp(min=1e-30)
    s_math = scale.to(wr.dtype)
    q = (wr / s_math).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale

# ---------------------------------------------------------------- numpy chain
def dequant_convrot_numpy(q_np, scale_np, gs):
    """Independent numpy-matmul chain (different accumulation order than
    torch; same spec)."""
    n, k = q_np.shape
    h = build_hadamard(gs, torch.float32).numpy()
    w = q_np.astype(np.float32) * scale_np
    wr = w.reshape(n, k // gs, gs) @ h.T
    return wr.reshape(n, k)

# ---------------------------------------------------------------- upstream import
def try_upstream():
    if not os.path.isdir(os.path.join(CK_PATH, "comfy_kitchen")):
        return None
    sys.path.insert(0, CK_PATH)
    try:
        from comfy_kitchen.backends.eager import quantization as qz  # noqa
        fn = qz.dequantize_int8_convrot_weight
        qq = qz.quantize_int8_convrot_weight
        return fn, qq
    except Exception as exc:  # pragma: no cover
        print(f"upstream import failed: {exc}", file=sys.stderr)
        return None

# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    h = load_header(FILE)
    sidecar_all = {}
    for e in h:
        if e.endswith(".comfy_quant"):
            raw, _, _ = tensor_bytes(FILE, h, e)
            # Comfy keys sidecars by stem: "<stem>.comfy_quant" where the
            # weight tensor is "<stem>.weight" (bare-stem spellings exist
            # for non-weight tensors; accept both).
            stem = e[: -len(".comfy_quant")]
            key = stem if stem.endswith(".weight") else stem + ".weight"
            sidecar_all[key] = json.loads(raw)
    upstream = try_upstream()
    meta = {}
    torch.set_num_threads(os.cpu_count())
    for name in TARGETS:
        qe = h[name]
        if qe["dtype"] != "I8":
            print(f"skip {name}: payload {qe['dtype']}")
            continue
        side = sidecar_all.get(name, {})
        if not (side.get("format") == "int8_tensorwise" and side.get("convrot")):
            print(f"skip {name}: sidecar {side}")
            continue
        gs = int(side["convrot_groupsize"])
        raw_q, _, shape = tensor_bytes(FILE, h, name)
        q = torch.frombuffer(bytearray(raw_q), dtype=torch.int8).view(*shape)
        n, k = shape
        raw_s, _, _ = tensor_bytes(FILE, h, name.replace(".weight", ".weight_scale"))
        scale = torch.frombuffer(bytearray(raw_s), dtype=torch.float32).view(n, 1)
        # --- reference chain (mirror)
        ref = dequant_convrot_mirrored(q, scale, gs)
        if upstream is not None:
            up = upstream[0](q, scale, gs)
            bit_equal = torch.equal(ref, up)
            if not bit_equal:
                bad = int((ref != up).sum())
                mx = float((ref - up).abs().max())
                raise SystemExit(f"mirror != upstream for {name}: {bad} elems, maxdiff {mx}")
        else:
            bit_equal = None
        # --- numpy independent chain
        npref = dequant_convrot_numpy(q.numpy(), scale.numpy(), gs)
        fp32_gap = float(np.max(np.abs(npref - ref.numpy()) / (np.abs(ref.numpy()) + 1e-12)))
        fp32_gap_scaled = float(np.max(np.abs(npref - ref.numpy()) / (np.abs(ref.numpy()).max(axis=1, keepdims=True) + 1e-12)))
        # --- bf16 cast (RNE, via torch)
        ref_bf16 = ref.to(torch.bfloat16).view(torch.uint16).numpy()
        # --- re-encode semantic check
        q2, s2 = quant_convrot_mirrored(ref.to(torch.bfloat16), gs)
        diff = (q2.to(torch.int16) - q.to(torch.int16)).abs()
        code_match = float((diff == 0).float().mean())
        code_within1 = float((diff <= 1).float().mean())
        scale_relerr = float(((s2 - scale).abs() / (scale + 1e-12)).max())
        # --- dumps
        stem = name.replace(".", "_")
        open(f"{OUT}/{stem}.q.bin", "wb").write(raw_q)
        open(f"{OUT}/{stem}.scale.bin", "wb").write(raw_s)
        ref.numpy().tofile(f"{OUT}/{stem}.ref_fp32.bin")
        ref_bf16.tofile(f"{OUT}/{stem}.ref_bf16.bin")
        npref.astype(np.float32).tofile(f"{OUT}/{stem}.np_fp32.bin")
        meta[name] = {
            "shape": [n, k], "groupsize": gs, "sidecar": side,
            "upstream_bit_equal": bit_equal,
            "mirror_vs_numpy_fp32_maxrel": fp32_gap,
            "mirror_vs_numpy_fp32_maxrel_rowScaled": fp32_gap_scaled,
            "requant_code_match_exact": code_match,
            "requant_code_match_within1": code_within1,
            "requant_scale_maxrelerr": scale_relerr,
        }
        print(f"{name}: gs={gs} upstream==mirror:{bit_equal} "
              f"requant exact={code_match:.4f} ±1={code_within1:.4f} "
              f"scale_rel={scale_relerr:.2e} np_gap_row={fp32_gap_scaled:.2e}")
    meta["_file"] = FILE
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(f"meta -> {OUT}/meta.json")

if __name__ == "__main__":
    main()
