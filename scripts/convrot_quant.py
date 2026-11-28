"""convrot_quant — canonical quantization line for h3c convrot conversion.

THE line (transcribed from scripts/convrot_ref.py, itself a bitwise-verified
mirror of upstream comfy_kitchen eager quantize_int8_convrot_weight; see
docs/evidence/2026-11-25_convrot三方驗收報告.json — upstream_bit_equal PASS):

  H_gs:  h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]   (CONFERENCE
         type, NOT Sylvester — never FWHT-substitute)
         kron-power to gs (gs a power of 4), / sqrt(gs)
  rotate: W.view(N, K//gs, gs) @ H_gs.T   in WEIGHT DTYPE (bf16)
  scale:  absmax over the FULL row of the rotated weight,
          scale = clamp((absmax.to(f32) / 127.0), min=1e-30)      [f32 tensor]
  divide: s_math = scale.to(bf16);  q = (rot / s_math)            [bf16 math]
          codes = clamp(round_RNE(q), -128, 127).to(int8)
  dequant mirror: (q.f32 * scale.f32) @ H_gs.T f32; H_gs/√gs is symmetric
          and orthogonal, so rotate is an involution: quant->dequant of an
          exact encoding returns the input up to int8 noise only.

Scale STORAGE (rowwise [N,1] vs grouped [N,K/gs]) is a file-format question,
not a math question; the builder copies the contract's scale bytes verbatim,
so it reproduces whichever spelling the official file uses.
"""
import numpy as np
import torch


class QuantError(ValueError):
    pass


H4_CACHE = {}


def build_hadamard(size, dtype=torch.float32):
    """Mirror of eager/convrot_w4a4.py _build_hadamard (kron power of h4)."""
    key = (size, dtype)
    if key in H4_CACHE:
        return H4_CACHE[key]
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
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


def rotate_stages_f32(x):
    """Rotate an already-reshaped (n, groups, gs) view with the f32 Hadamard.

    Restored 2026-11-28 (scripts/ drift: scripts/qkv_layout_gate.py imports
    this single-argument form; the builder path keeps rotate_weight). Same
    group math as rotate_weight, done in f32 (gate recompute convention).
    """
    gs = x.shape[-1]
    h = build_hadamard(gs, torch.float32)
    return (x.to(torch.float32) @ h.T).reshape(x.shape)


def qkv_grouped_perm(rows):
    """Permute interleaved-HF rows -> kind-grouped file rows (1-arg form).

    Restored 2026-11-28 for scripts/qkv_layout_gate.py's import contract;
    semantics identical to build_int8_convrot.py's local (rows, n_rows) twin,
    empirically pinned by scratch/sol/probe_qkv_layout.py (official file
    stores qkv rows kind-grouped; HF reference interleaved). Works on torch
    or numpy row arrays; n_rows = rows.shape[0].
    """
    n = rows.shape[0]
    if n % 384:
        raise QuantError(f"qkv rows {n} not divisible by 384")
    hpc = n // 384
    j = np.arange(n)
    kind, unit, sub = j // (hpc * 128), (j // 128) % hpc, j % 128
    src = (unit * 3 + kind) * 128 + sub
    if hasattr(rows, "detach"):  # torch tensor: index with a torch tensor
        src = torch.from_numpy(src)
    return rows[src]


def quant_tensor_f32line(w, gs: int = 256, mode: str = "int8"):
    """Quantize one weight the upstream convrot line (mode='int8').

    w: torch tensor [N, K]; non-bf16 input is cast to bf16 first (HF canonical
       storage is bf16 for all quant subjects — asserted by the builder).
    gs: convrot groupsize (power of 4: 16?/64/256; upstream h4 base means the
        real file uses 64 and 256; 16 falls back to h4 kron floor = 4? guarded
        below: sizes <4 unsupported).
    mode='matmul_ref': rotate in f32 instead of bf16 (diagnostic upper bound
        only — NOT the contract; codes may differ from int8 mode at ulp level
        and the returned scale/divide math is identical otherwise).
    Returns (codes int8 torch [N,K], scale f32 torch [N]).
    """
    if w.ndim != 2:
        raise QuantError(f"expected 2D, got {tuple(w.shape)}")
    n, k = w.shape
    if gs < 4 or gs & (gs - 1):
        raise QuantError(f"convrot groupsize {gs} must be a power of two >= 4")
    if gs & 2:  # conference h4 base: size must be a power of 4
        raise QuantError(f"convrot groupsize {gs} must be a power of 4 (h4 base)")
    if k % gs:
        raise QuantError(f"inner dim {k} not divisible by groupsize {gs}")

    if mode == "int8":
        w = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
        h = build_hadamard(gs, torch.bfloat16)
        wr = rotate_weight(w, h, gs)
    elif mode == "matmul_ref":
        h = build_hadamard(gs, torch.float32)
        wr = rotate_weight(w.float(), h, gs)
    else:
        raise QuantError(f"unknown mode {mode!r}")

    absmax = wr.abs().amax(dim=-1, keepdim=True)
    scale = (absmax.float() / 127.0).clamp(min=1e-30)     # f32 storage scale
    s_math = scale.to(wr.dtype)                            # bf16 divide math
    q = (wr / s_math).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def dequant_f32(codes_i8, scale_f32, gs: int):
    """Upstream dequantize_int8_convrot_weight mirror (f32).

    codes_i8: [N,K] int8 (torch or numpy), scale_f32: [N,1] or [N].
    Returns f32 torch tensor [N,K] in the ORIGINAL (unrotated) basis.
    """
    q = torch.from_numpy(np.asarray(codes_i8)).to(torch.int8) \
        if not torch.is_tensor(codes_i8) else codes_i8.to(torch.int8)
    s = torch.from_numpy(np.asarray(scale_f32, dtype=np.float32)) \
        if not torch.is_tensor(scale_f32) else scale_f32.to(torch.float32)
    if s.ndim == 1:
        s = s.view(-1, 1)
    n, k = q.shape
    h = build_hadamard(gs, torch.float32)
    return rotate_weight(q.float() * s, h, gs)
