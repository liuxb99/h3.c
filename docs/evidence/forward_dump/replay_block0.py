#!/usr/bin/env python3
"""Eager block-0 replay vs h3.c Metal engine dumps (H3_DUMP_DIR tap set).

Engine constants (h3_dit.c enum @ HEAD 8080d74):
  HIDDEN=5376 HEADS=56 HEAD_DIM=128 INNER=7168 FFN=14336 ROPE_HALF=48 SLOTS=6
  eps=1e-5; attention scale 1/sqrt(128); block linears bias-free (bias folded
  into the adaln table at load); qkv buffer + qkv weight in grouped per-head
  layout (q|k|v 128 each per head block); adaLN/gate/swiglu/linear each round
  to bf16 exactly once at the end of their f32 expression (verified against
  h3_shaders.metal kernels h3_adaln_bf16, h3_gate_adaln_bf16,
  h3_qkv_rope_bf16 grouped=1, h3_swiglu_bf16, h3_linear_f32_tiled_bf16).

Gate: >0.5 bf16 ulp (at the engine value's exponent) counts as a divergence
hit. Also verifies RoPE table provenance vs b0_layout_positions (t,h,w).
"""
import glob
import os

import numpy as np
import torch

DUMP = "/Users/cyliu99/.hermes/cache/scratch/forward_dump"
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
ROPE_HALF = 48
EPS = 1e-5
SCALE = 1.0 / (HEAD_DIM ** 0.5)

results = []


def _find(tag):
    hits = sorted(glob.glob(os.path.join(DUMP, tag + "_*.raw")))
    return hits[0] if hits else None


def load_raw(tag):
    """Return (tensor, dtype_tag) for a tap, or None."""
    p = _find(tag)
    if p is None:
        return None
    dtype = os.path.basename(p)[: -len(".raw")].rsplit("_", 1)[-1]
    raw = open(p, "rb").read()
    if dtype == "f32":
        return torch.tensor(np.frombuffer(raw, dtype="<f4").copy(), dtype=torch.float32), dtype
    if dtype == "f64":
        return torch.tensor(np.frombuffer(raw, dtype="<f8").copy(), dtype=torch.float64), dtype
    if dtype == "u32":
        return torch.tensor(np.frombuffer(raw, dtype="<u4").copy(), dtype=torch.int64), dtype
    if dtype == "i8":
        return torch.tensor(np.frombuffer(raw, dtype="<i1").copy(), dtype=torch.int8), dtype
    return torch.from_numpy(np.frombuffer(raw, dtype="<u2").copy()).view(torch.bfloat16), dtype


def as_rows(tag, cols):
    r = load_raw(tag)
    if r is None:
        return None
    t = r[0]
    if t.numel() % cols:
        raise SystemExit(f"{tag}: numel {t.numel()} not divisible by cols {cols}")
    return t.reshape(-1, cols)


def as_vec(tag):
    r = load_raw(tag)
    return None if r is None else r[0].reshape(-1)


def compare(tag, eager):
    eng = as_rows(tag, eager.shape[-1]) if eager is not None else None
    if eng is None:
        results.append((tag, "NO ENGINE TAP", "-", "-", "-"))
        return
    if eng.shape != eager.shape:
        results.append((tag, f"SHAPE {tuple(eng.shape)} vs {tuple(eager.shape)}", "-", "-", "-"))
        return
    e = eng.float().double().numpy()
    a = eager.detach().float().double().numpy()
    ulp = np.ldexp(np.ones_like(e), (np.floor(np.log2(np.maximum(np.abs(e), 1e-30))) - 7).astype(np.int32))
    steps = np.abs(e - a) / ulp
    steps = np.where(e == 0.0, 0.0, steps)  # engine exact-zero: ulp meaningless there
    mx = float(np.max(steps))
    n_bad = int(np.sum(steps > 0.5))
    results.append((tag, f"max {mx:.2f} ulp", f"{n_bad}/{e.size}",
                    f"{np.abs(e - a).max():.4g}", f"{np.abs(e).max():.4g}"))


def main():
    hidden = as_rows("b0_hidden_in", HIDDEN)
    rows = hidden.shape[0]
    mod = as_rows("b0_adaln_mod_table", 6 * HIDDEN)
    r = load_raw("b0_row_map")
    if r is not None:
        rm = r[0].reshape(-1)
        if rm.numel() == rows and int(rm.max()) < mod.shape[0]:
            mod = mod[rm]
        elif rm.numel() != rows:
            print(f"note: row_map numel {rm.numel()} != rows {rows}; using identity")
    qkv_w = as_rows("b0_qkv_w", HIDDEN)
    q_norm_w = as_vec("b0_q_norm_w")
    k_norm_w = as_vec("b0_k_norm_w")
    norm1_w = as_vec("b0_norm1_w")
    norm2_w = as_vec("b0_norm2_w")
    out_w = as_rows("b0_out_w", INNER)
    fc1_w = as_rows("b0_fc1_w", HIDDEN)
    fc2_w = as_rows("b0_fc2_w", FFN)
    cos = load_raw("b0_rope_cos_bf16")[0]
    sin = load_raw("b0_rope_sin_bf16")[0]
    if cos.numel() % rows:
        raise SystemExit(f"rope table numel {cos.numel()} not rows*48")
    cos = cos.reshape(rows, ROPE_HALF)
    sin = sin.reshape(rows, ROPE_HALF)

    h32 = hidden.float()

    def rmsnorm_scale(x32, w):
        inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return x32 * inv * w.float().reshape(1, -1)

    sh_a, sc_a, gt_a = mod[:, :HIDDEN], mod[:, HIDDEN:2 * HIDDEN], mod[:, 2 * HIDDEN:3 * HIDDEN]
    sh_m, sc_m, gt_m = (mod[:, 3 * HIDDEN:4 * HIDDEN], mod[:, 4 * HIDDEN:5 * HIDDEN],
                        mod[:, 5 * HIDDEN:])

    h_norm = rmsnorm_scale(h32, norm1_w)
    attn_in = (h_norm * (1 + sc_a.float()) + sh_a.float()).to(torch.bfloat16)
    compare("b0_attn_adaln_out", attn_in)

    qkv = (attn_in.float() @ qkv_w.float().t()).to(torch.bfloat16)
    compare("b0_qkv_pre_norm", qkv)

    g = qkv.float().reshape(rows, HEADS, 3 * HEAD_DIM)
    q, k, v = g[:, :, :HEAD_DIM], g[:, :, HEAD_DIM:2 * HEAD_DIM], g[:, :, 2 * HEAD_DIM:]

    def qknorm_rope(t, wvec):
        tn = rmsnorm_scale(t, wvec)
        c = cos.float().reshape(rows, 1, ROPE_HALF)
        s = sin.float().reshape(rows, 1, ROPE_HALF)
        lo, hi = tn[..., :ROPE_HALF], tn[..., ROPE_HALF:2 * ROPE_HALF]
        return torch.cat([lo * c - hi * s, hi * c + lo * s, tn[..., 2 * ROPE_HALF:]],
                         dim=-1).to(torch.bfloat16)

    q_hat = qknorm_rope(q, q_norm_w)
    k_hat = qknorm_rope(k, k_norm_w)
    compare("b0_q_post_qknorm_rope", q_hat.reshape(rows, INNER))
    compare("b0_k_post_qknorm_rope", k_hat.reshape(rows, INNER))
    compare("b0_v_after_qkv", v.reshape(rows, INNER))

    qh = q_hat.float().reshape(rows, HEADS, HEAD_DIM).permute(1, 0, 2)
    kh = k_hat.float().reshape(rows, HEADS, HEAD_DIM).permute(1, 0, 2)
    vh = v.permute(1, 0, 2)
    scores = torch.softmax(qh @ kh.transpose(-1, -2) * SCALE, dim=-1)
    attn = (scores @ vh).permute(1, 0, 2).reshape(rows, INNER)
    compare("b0_attention_out", attn.to(torch.bfloat16))

    outp = (attn @ out_w.float().t()).to(torch.bfloat16)
    compare("b0_out_proj_out", outp)

    gated = (h32 + outp.float() * gt_a.float()).to(torch.bfloat16)
    compare("b0_post_attn_hidden", gated)
    g32 = gated.float()
    fc1_in = (rmsnorm_scale(g32, norm2_w) * (1 + sc_m.float()) + sh_m.float()).to(torch.bfloat16)
    compare("b0_mlp_adaln_in", fc1_in)

    fc1 = (fc1_in.float() @ fc1_w.float().t()).to(torch.bfloat16)
    compare("b0_fc1_out", fc1)

    gate = fc1[:, :FFN].float()
    up = fc1[:, FFN:].float()
    act = (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16)
    compare("b0_swiglu_out", act)

    fc2 = (act.float() @ fc2_w.float().t()).to(torch.bfloat16)
    compare("b0_fc2_out", fc2)

    block_out = (g32 + fc2.float() * gt_m.float()).to(torch.bfloat16)
    compare("b0_block_out", block_out)

    # RoPE table provenance (prepare_rope): per row, axes = (t, h*scale, w*scale),
    # table[row][axis*16 + f] = cos/sin(axis_value_f32 * inv_freq[f]); the engine
    # bakes this f32 table into the bf16 GPU table used by the rope kernel.
    pos = load_raw("b0_layout_positions")
    inv = load_raw("b0_rope_inv_freq")
    if pos is None:
        results.append(("rope_table_vs_positions", "NO POSITIONS TAP", "-", "-", "-"))
    else:
        P = pos[0].reshape(rows, 3).to(torch.float32)
        if inv is None:  # read from the model shard if not dumped
            import json, struct
            p = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/FL2VA/transformer/model-00001-of-00013.safetensors"
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
                off = hdr["rope.inv_freq"]["data_offsets"][0]
                f.seek(8 + n + off)
                inv = torch.tensor(np.frombuffer(f.read(64), dtype="<f4").copy())
        scale = 1.0  # 64x64 render: spatial_rope_scale = 1.0 (0.5 only at 256)
        axes = torch.stack([P[:, 0], P[:, 1] * scale, P[:, 2] * scale], dim=1)
        ang = (axes.reshape(rows, 3, 1) * inv.reshape(1, 1, 16)).reshape(rows, 48)
        eager_cos = torch.cos(ang).to(torch.bfloat16)
        eager_sin = torch.sin(ang).to(torch.bfloat16)
        compare("b0_rope_cos_table", torch.cos(ang).reshape(rows, ROPE_HALF))
        compare("b0_rope_sin_table", torch.sin(ang).reshape(rows, ROPE_HALF))
        for nm, eag in (("b0_rope_cos_bf16", eager_cos), ("b0_rope_sin_bf16", eager_sin)):
            compare(nm, eag.reshape(rows, ROPE_HALF))

    print(f"rows={rows}")
    w = max(len(x[0]) for x in results) + 1
    for r in results:
        if len(r) == 4:  # provenance rows
            print(f"{r[0]:<{w}} {r[1]:<30}")
        else:
            print(f"{r[0]:<{w}} {r[1]:<17} hits: {r[2]:<12} maxabs: {r[3]:<9} engine |max|: {r[4]}")


if __name__ == "__main__":
    import torch as _t
    _t.set_num_threads(10)
    main()
