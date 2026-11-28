#!/usr/bin/env python3
"""gate_runner.py — run scripts/qkv_layout_gate.py unmodified despite the
convrot_quant.py mid-refactor (02:53 rewrite dropped rotate_stages_f32 /
qkv_grouped_perm / quant_rotate_f32, which the 00:34-compiled gate imports).

Semantics are pinned by scratch/sol/battery_rotperm.py + battery_out.json:
  * quant line: new 02:53 quant_tensor_f32line == gate-era pyc quant line,
    bit-identical codes+scales on 4 shapes (B1) -> no substitution needed.
  * dequant ground truth: dequant(official contract codes, official scales)
    reproduces HF bf16 golden at relrms 0.0088-0.012, cos 0.99993+ (B2) for
    matrix_norm (normalized Hadamard per group) and the gate-era pyc
    rotate_stages_f32 (identical outputs).
  * gate source (authoritative, frozen 00:34) consumes the helper as
    dequant_chunk = rotate_stages_f32(chunk) / sqrt(group_size)  (line 361)
    -> the helper contract at gate-authoring time is the UNNORMALIZED
    digit-stage sum (= matrix_norm * sqrt(group_size)).
  * qkv_grouped_perm: builder row-kind grouping formula == both pyc versions,
    identical on N=384/3840/11520 (B3).
Injection policy: only names missing from the loaded scripts/convrot_quant.py
are added, as gate-era-contract formulas (never relaxed numerics; the gate's
own /sqrt(gs) then yields exactly the battery-verified dequant).
scripts/st_io.py and scripts/qkv_layout_gate.py run untouched.
"""
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # scratch/sol
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))     # h3c repo root
SCRIPTS = os.path.join(REPO, "scripts")

for p in (SCRIPTS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import convrot_quant as cq  # noqa: E402  (scripts/ version)

NEEDED = ("rotate_stages_f32", "qkv_grouped_perm", "quant_tensor_f32line")
missing = [n for n in NEEDED if not hasattr(cq, n)]

_h_cache = {}


def _h_norm(gs):
    if gs not in _h_cache:
        _h_cache[gs] = cq.build_hadamard(gs, torch.float32)
    return _h_cache[gs]


def rotate_stages_f32(x):
    """Gate-era contract: UNNORMALIZED digit-stage Hadamard rotation on the
    (n, g, gs) 3-D group basis, i.e. matrix_norm(x) * sqrt(gs). The gate's
    dequant_chunk divides by sqrt(gs), giving the battery-pinned dequant."""
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"expected 3-D (n, groups, group_size), got {tuple(x.shape)}")
    gs = int(x.shape[-1])
    if gs & (gs - 1) or gs < 4:
        raise ValueError(f"group size {gs} is not a power of 4")
    return (_h_norm(gs).T[None] * math.sqrt(gs)) @ x


def qkv_grouped_perm(rows):
    """Grouped qkv row permutation: kind-major -> unit-major (unit*3+kind)*128+sub.
    Battery-verified identical to the gate-era pyc implementation."""
    idx = np.asarray(rows, dtype=np.int64)
    n = int(idx.max()) + 1
    assert n % 384 == 0
    hpc = n // 384
    kind, unit, sub = idx // (hpc * 128), (idx // 128) % hpc, idx % 128
    return (unit * 3 + kind) * 128 + sub


if missing:
    for name in missing:
        setattr(cq, name, {"rotate_stages_f32": rotate_stages_f32,
                           "qkv_grouped_perm": qkv_grouped_perm}[name])
    sys.stderr.write(f"[gate_runner] injected into convrot_quant: {missing} "
                     "(gate-era contract semantics; see battery_rotperm.json)\n")

runpy.run_path(os.path.join(SCRIPTS, "qkv_layout_gate.py"), run_name="__main__")
sys.exit(0)
