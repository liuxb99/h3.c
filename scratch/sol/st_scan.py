"""Shared helpers for fused-projection row-order scan (stdlib-only).

Hand-parse rules (skill h3c-native-video-engine / docs/design convrot 定案):
  - data section starts at byte 8 + len(header_json)
  - data_offsets in header entries are RELATIVE to data start
  - bf16 decode: uint16 -> uint32<<16 -> float32 view
"""
import json
import mmap
import numpy as np
import re
import struct

FUSED_RE = re.compile(r"qkv|kv_proj|to_q|to_k|to_v|q_proj|k_proj|v_proj|kv|fc1|w1$")


def open_st(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"f": f, "mm": mm, "hdr": hdr, "base": base, "path": path}


def raw(st, name, r0=0, r1=None):
    """Return bytes of tensor rows [r0:r1) (row-major last-dim contiguous)."""
    e = st["hdr"][name]
    shp = e["shape"]
    dt = e["dtype"]
    itemsize = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8,
                "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}[dt]
    cols = int(np.prod(shp[1:])) if len(shp) > 1 else 1
    b0, b1 = e["data_offsets"]
    b0 += r0 * cols * itemsize
    b1 = b0 + ((r1 if r1 is not None else shp[0]) - r0) * cols * itemsize
    return st["mm"][st["base"] + b0: st["base"] + b1]


def deq_bf16(st, name, r0=0, r1=None):
    b = np.frombuffer(raw(st, name, r0, r1), dtype=np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32).reshape(
        (r1 if r1 is not None else st["hdr"][name]["shape"][0]) - r0, -1)


def deq_int8_scaled(st, name, scale_name):
    """convrot / tensorwise int8: codes I8 x per-row scale -> rotated-row f32."""
    codes = np.frombuffer(raw(st, name), dtype=np.int8).astype(np.float32)
    codes = codes.reshape(st["hdr"][name]["shape"])
    scale = deq_sidecar_scale(st, scale_name)  # [rows,1] f32
    assert scale.shape[0] == codes.shape[0], (scale.shape, codes.shape)
    return codes * scale


def deq_sidecar_scale(st, scale_name):
    e = st["hdr"][scale_name]
    dt = e["dtype"]
    b = np.frombuffer(bytes(st["mm"][st["base"] + e["data_offsets"][0]:
                            st["base"] + e["data_offsets"][1]]), dtype=np.uint8)
    if dt == "F32":
        v = b.view(np.float32)
    elif dt == "BF16":
        v = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F8_E4M3":
        v = fp8_e4m3_to_f32(b)
    else:
        raise ValueError("scale dtype " + dt)
    return v.reshape(e["shape"][0], -1)


def fp8_e4m3_to_f32(b):
    """byte-exact software cast for f8_e4m3fn (no subnormal worry for scales>0)."""
    b = b.view(np.uint8).astype(np.uint32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    man = b & 7
    # normal: value = 2^(e-7) * (1 + m/8); e=0 subnormal: 2^-6 * m/8
    e = np.where(exp == 0, 1, exp)  # unbiased+127 trick below
    normal = exp != 0
    unbiased = np.where(normal, exp.astype(np.int32) - 7, -6)
    mant = np.where(normal, 1.0 + man / 8.0, man / 8.0)
    v = np.ldexp(mant.astype(np.float64), unbiased).astype(np.float32)
    return np.where(sign, -v, v)


def hadamard(gs):
    """H_gs = h4^{otimes log4(gs)} / sqrt(gs), conference h4 (docs/design 定案)."""
    h4 = np.array([[1, 1, 1, -1],
                   [1, 1, -1, 1],
                   [1, -1, 1, 1],
                   [-1, 1, 1, 1]], dtype=np.float32)
    h = np.array([[1]], dtype=np.float32)
    while h.shape[0] < gs:
        h = np.kron(h, h4)
    return (h / np.sqrt(gs)).astype(np.float32)


def rotate_rows(x, H):
    """groups run along K contiguous (per row, block-diag H over last dim)."""
    g = H.shape[0]
    r, c = x.shape
    assert c % g == 0
    return (x.reshape(r, c // g, g) @ H.T).reshape(r, c)


def cos_pairs(A, B):
    """full-matrix cosine between two [R,C] float arrays in given row order."""
    A = A.astype(np.float64).ravel(); B = B.astype(np.float64).ravel()
    return float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B)))


# --- row-order transforms (out dim = 3*heads*head_dim; 384-row head block) ---
def interleave_from_grouped(W):
    """W stored [allQ;allK;allV] (kind-major 7168-row slabs) -> per-head
    interleaved 384-row blocks [Q_h(128);K_h(128);V_h(128)] (engine/HF order)."""
    rows, cols = W.shape
    assert rows % 384 == 0
    blocks = rows // 384          # heads_per_component (=56)
    W3 = W.reshape(3, blocks, 128, cols)   # [kind, head, d, cols]
    return W3.transpose(1, 0, 2, 3).reshape(rows, cols)  # [head, kind, d] flat


def grouped_from_interleaved(W):
    """Inverse: per-head 384-row blocks -> [allQ;allK;allV]."""
    rows, cols = W.shape
    assert rows % 384 == 0
    blocks = rows // 384
    return W.reshape(blocks, 3, 128, cols).transpose(1, 0, 2, 3).reshape(rows, cols)


def fused_keys(st):
    out = []
    for k, e in st["hdr"].items():
        if k.endswith(".comfy_quant") or k.endswith("weight_scale") or k.endswith("scale"):
            continue
        if FUSED_RE.search(k):
            out.append((k, e["dtype"], e["shape"]))
    return out
