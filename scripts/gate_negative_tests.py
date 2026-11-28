#!/usr/bin/env python3
"""gate_negative_tests.py — negative-rejection battery for scripts/qkv_layout_gate.py.

Builds a small SYNTHETIC convrot model (golden seed, gaussian weights) plus
five single-point-corruption variants, runs the gate on each against a bf16
HF-order reference with the gate's DEFAULT thresholds (never loosened), and
asserts:

  CONTROL  correct synthetic model   -> exit 0, verdict PASS
  M1 qkv layout wrong (inter order)  -> reject with FAIL [L] evidence on qkv
  M2 sidecar scale x2 on one subject -> reject with FAIL [Q] relrms evidence
  M3 double Hadamard (codes quantized from the UNROTATED weight, i.e. the
     dequant lands one extra Hadamard off the reference basis) -> FAIL [Q]
  M4 0.1% of one subject's int8 code bytes bit-flipped          -> FAIL [Q]
  M5 comfy_quant sidecar deleted for one subject (subject then
     degrades to a pass-through key whose I8 payload cannot match the BF16
     reference) -> FAIL [P] evidence

A variant that the gate fails to reject = GAP. The script writes
scratch/sol/gate_neg/matrix.txt (+ per-run gate logs/detail JSON in
scratch/sol/gate_neg/logs/) and copies the matrix to the evidence folder.
Sandbox dir is wiped and rebuilt at every start (idempotent). Exit 0 iff all
six rows behave as expected.

KNOWN GATE-TRIAGE QUIRKS (observed, verdicts unaffected — gate left unmodified
per task discipline): qkv_layout_gate.classify_permutation reshapes the flat
code payload to [N,K] and then np.sort-sorts ROW-WISE, not as one global
multiset, and requant_ref pre-permutes the reference codes to grouped order;
so the diagnostic tag is unreliable for the cases it labels — M1 (row
permutation) tags QUANT_DRIFT (true global-multiset diff is 0.0 => should be
PERMUTATION), M2 (scale error, codes untouched) tags PERMUTATION. The FAIL
verdicts and numeric measurements are correct regardless.

Run (CPU only):
  cd <repo root> && uv run --python 3.11 --with torch,numpy -q python \
      scripts/gate_negative_tests.py

NOTE: scripts/st_io.py's st_create currently references an undefined `build`
(NameError on any call — scripts/ drift, outside this test's edit scope), so
this battery ships its own minimal standard-spelling safetensors writer
(8-byte u64 header length, space-padded header so the data section starts
64-aligned, DATA-relative offsets, 64-aligned payload starts, no 'offset'
key). The gate's st_open reader consumes it natively.
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, SCRIPTS)

from convrot_quant import quant_tensor_f32line, qkv_grouped_perm  # noqa: E402

GATE = os.path.join(SCRIPTS, "qkv_layout_gate.py")
SBX = os.path.join(REPO, "scratch", "sol", "gate_neg")
LOGS = os.path.join(SBX, "logs")
MATRIX = os.path.join(SBX, "matrix.txt")
EVIDENCE = os.path.join(REPO, "..", "evidence", "convrot_audit_20261128",
                        "gate_negative_matrix.txt")

ALIGN = 64
SEED = 20261128


# ---------------------------------------------------------------- safetensors
def st_write(path, tensors):
    """Minimal standard-spelling safetensors writer.

    tensors: list of (name, dtype_str, shape_list, payload_bytes), any order;
    payload starts 64-aligned both data-relative and file-absolute.
    """
    offs, rel = {}, 0
    for name, _dt, _sh, pb in tensors:
        offs[name] = rel
        rel += len(pb)
        rel = -(-rel // ALIGN) * ALIGN
    hdr = {name: {"dtype": dt, "shape": list(sh),
                  "data_offsets": [offs[name], offs[name] + len(pb)]}
           for name, dt, sh, pb in tensors}
    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-(8 + len(raw))) % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        assert (8 + len(raw)) % ALIGN == 0
        prev = 0
        for name, _dt, _sh, pb in tensors:
            if offs[name] > prev:
                f.write(b"\0" * (offs[name] - prev))
            f.write(pb)
            prev = offs[name] + len(pb)


# ------------------------------------------------------------------ synthetic
# quant subjects: name -> (N, K, groupsize); qkv names carry the fused
# [Q|K|V] kind-grouped storage order the convrot contract pins (gate
# --expect-layout grouped default; N must be a multiple of 384).
SUBJECTS = [
    ("blocks.0.attn.qkv_proj.weight", 768, 512, 256),
    ("blocks.0.attn.out_proj.weight", 256, 512, 256),
    ("blocks.0.mlp.fc1.weight", 1024, 512, 256),
    ("blocks.0.mlp.fc2.weight", 512, 1024, 256),
    ("blocks.0.adaln_proj.linear.weight", 576, 512, 64),
    ("blocks.1.attn.qkv_proj.weight", 768, 512, 256),
    ("blocks.1.attn.out_proj.weight", 256, 512, 256),
    ("blocks.1.mlp.fc1.weight", 1024, 512, 256),
    ("blocks.1.mlp.fc2.weight", 512, 1024, 256),
    ("blocks.1.adaln_proj.linear.weight", 576, 512, 64),
    ("final_layer.linear.weight", 512, 512, 256),
    ("time_mod.mlp.0.weight", 256, 512, 256),
    ("time_mod.mlp.2.weight", 512, 256, 256),
]
# adaLN padding: trailing zero rows must quantize to scale==1e-30 + zero
# codes (gate [S0] rule).
ADALN_ZERO_ROWS = 32
# pass-through BF16 keys (>=3): byte-identical in candidate and reference.
PT_KEYS = [
    ("blocks.0.norm.weight", [512]),
    ("blocks.0.norm.bias", [512]),
    ("blocks.1.norm.weight", [512]),
    ("pos_embed", [1, 512]),
]
QKV0 = "blocks.0.attn.qkv_proj.weight"
M2_SUBJ = "blocks.0.attn.out_proj.weight"
M3_SUBJ = "blocks.0.mlp.fc1.weight"
M4_SUBJ = "blocks.0.mlp.fc2.weight"
M5_SUBJ = "blocks.1.attn.out_proj.weight"


def quant_unrotated(w_bf16, gs):
    """M3: the pinned f32 scale line applied to the UNROTATED weight — codes
    that make the gate's dequant (which applies H once) land in the rotated
    basis, i.e. the effect of applying the Hadamard one time too many."""
    wr = w_bf16  # deliberately no rotation
    absmax = wr.float().abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 127.0).clamp(min=1e-30)
    s_math = scale.to(wr.dtype)
    codes = (wr / s_math).round().clamp(-128.0, 127.0).to(torch.int8)
    return codes.numpy(), scale.numpy()


def build_model():
    """Return (ref_tensors, cand_base, aux).

    ref_tensors  : ordered (name, dtype, shape, bytes) — bf16 HF-order golden.
    cand_base    : dict key -> (dtype, shape, bytes) for the correct candidate
                   (qkv subjects stored kind-grouped per the contract).
    aux          : subject -> (codes_hf [N,K] i8, scale_hf [N,1] f32, gs) for
                   the variant builders.
    """
    gen = torch.Generator().manual_seed(SEED)
    ref_tensors, cand_base, aux = [], {}, {}
    for name, N, K, gs in SUBJECTS:
        W = torch.randn(N, K, generator=gen)
        if "adaln" in name:
            W[-ADALN_ZERO_ROWS:, :] = 0.0
        Wb = W.to(torch.bfloat16)
        ref_tensors.append((name, "BF16", [N, K],
                            Wb.view(torch.uint16).numpy().tobytes()))
        codes, scale = quant_tensor_f32line(Wb, gs, mode="int8")
        codes, scale = codes.numpy(), scale.numpy()
        aux[name] = (codes.copy(), scale.copy(), gs)
        if name.endswith("attn.qkv_proj.weight") and N % 384 == 0:
            cg, sg = qkv_grouped_perm(codes), qkv_grouped_perm(scale)
        else:
            cg, sg = codes, scale
        cand_base[name] = ("I8", [N, K], cg.tobytes())
        cand_base[name + ".weight_scale"] = ("F32", [N, 1], sg.tobytes())
        cj = json.dumps({"format": "int8_tensorwise", "convrot": True,
                         "convrot_groupsize": gs},
                        separators=(",", ":")).encode("utf-8")
        cand_base[name + ".comfy_quant"] = ("U8", [len(cj)], bytes(cj))
    for name, shape in PT_KEYS:
        t = torch.randn(*shape, generator=gen).to(torch.bfloat16)
        pb = t.view(torch.uint16).numpy().tobytes()
        ref_tensors.append((name, "BF16", shape, pb))
        cand_base[name] = ("BF16", shape, pb)
    return ref_tensors, cand_base, aux


def write_candidate(path, cand_base, tensors_extra_order=None):
    tn = [(k, dt, sh, pb) for k, (dt, sh, pb) in cand_base.items()]
    st_write(path, tn)


def variant_m1(cand_base, aux):
    """QKV rows written in inter/HF order instead of kind-grouped."""
    codes_hf, scale_hf, _ = aux[QKV0]
    cand_base[QKV0] = ("I8", list(codes_hf.shape), codes_hf.tobytes())
    cand_base[QKV0 + ".weight_scale"] = ("F32", list(scale_hf.shape),
                                         scale_hf.tobytes())


def variant_m2(cand_base):
    """Scale sidecar x2 on one subject (codes untouched)."""
    dt, sh, pb = cand_base[M2_SUBJ + ".weight_scale"]
    s = np.frombuffer(pb, dtype=np.float32) * np.float32(2.0)
    cand_base[M2_SUBJ + ".weight_scale"] = (dt, sh, s.tobytes())


def variant_m3(cand_base, aux_ref_W, ref_by_name):
    """Codes/scale requantized from the UNROTATED weight (double Hadamard)."""
    name = M3_SUBJ
    Wb = aux_ref_W[name]
    codes, scale = quant_unrotated(Wb, ref_by_name[name][2])
    cand_base[name] = ("I8", list(codes.shape), codes.tobytes())
    cand_base[name + ".weight_scale"] = ("F32", list(scale.shape),
                                         scale.tobytes())


def variant_m4(cand_base, seed=4242):
    """Flip random bits in 0.1% of one subject's int8 code bytes."""
    dt, sh, pb = cand_base[M4_SUBJ]
    rng = np.random.default_rng(seed)
    a = bytearray(pb)
    n = len(a)
    idx = rng.choice(n, size=max(1, n // 1000), replace=False)
    bits = rng.integers(0, 8, size=len(idx))
    for i, b in zip(idx, bits):
        a[i] ^= np.uint8(1) << np.uint8(b)
    cand_base[M4_SUBJ] = (dt, sh, bytes(a))


def variant_m5(cand_base):
    """Delete one subject's comfy_quant sidecar entirely."""
    del cand_base[M5_SUBJ + ".comfy_quant"]


# ------------------------------------------------------------------ gate runs
def run_gate(case, cand_path, ref_path):
    logp = os.path.join(LOGS, f"{case}.gate.log")
    detp = os.path.join(LOGS, f"{case}.detail.json")
    cmd = [sys.executable, GATE,
           "--candidate", cand_path,
           "--reference", ref_path,
           "--detail-out", detp,
           "--allow-missing-reference-keys"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=REPO)
    out = p.stdout + "\n" + p.stderr
    with open(logp, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n" + out)
    m = re.search(r"SUMMARY (\{.*\})", out)
    summary = json.loads(m.group(1)) if m else {}
    return p.returncode, out, summary


def main():
    # sandbox: wipe & rebuild on every start (idempotent)
    if os.path.isdir(SBX):
        shutil.rmtree(SBX)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)

    ref_tensors, cand_base, aux = build_model()
    ref_by_name = {t[0]: t for t in ref_tensors}
    aux_ref_W = {}
    for name, N, K, gs in SUBJECTS:
        _n, _dt, _sh, pb = ref_by_name[name]
        w_f32 = (np.frombuffer(pb, dtype=np.uint16).astype(np.uint32)
                 << 16).view(np.float32)  # bf16 -> f32 is bit-exact
        aux_ref_W[name] = torch.from_numpy(w_f32).to(torch.bfloat16).reshape(N, K)

    ref_path = os.path.join(SBX, "reference_bf16.safetensors")
    st_write(ref_path, ref_tensors)

    cases = []  # (case, expect, evidence_regex, mutator)
    cases.append(("CONTROL_correct", "accept", None, None))
    cases.append(("M1_qkv_inter_order", "reject",
                  rf"FAIL \[L\].*{re.escape(QKV0)}.*expect=grouped",
                  lambda cb: variant_m1(cb, aux)))
    cases.append(("M2_scale_x2", "reject",
                  rf"FAIL \[Q\].*{re.escape(M2_SUBJ)}.*relrms",
                  lambda cb: variant_m2(cb)))
    cases.append(("M3_double_hadamard", "reject",
                  rf"FAIL \[Q\].*{re.escape(M3_SUBJ)}",
                  lambda cb: variant_m3(cb, aux_ref_W, ref_by_name)))
    cases.append(("M4_codes_bitflip", "reject",
                  rf"FAIL \[Q\].*{re.escape(M4_SUBJ)}",
                  lambda cb: variant_m4(cb)))
    cases.append(("M5_missing_comfy_quant", "reject",
                  rf"FAIL \[P\].*{re.escape(M5_SUBJ)}",
                  lambda cb: variant_m5(cb)))

    rows, gaps = [], []
    for case, expect, pat, mut in cases:
        cand_path = os.path.join(SBX, f"{case}.safetensors")
        cb = {k: (dt, sh, bytes(pb)) for k, (dt, sh, pb) in cand_base.items()}
        if mut is not None:
            mut(cb)
        write_candidate(cand_path, cb)
        rc, out, summary = run_gate(case, cand_path, ref_path)
        ev_lines = [l.strip() for l in out.splitlines() if "FAIL [" in l]
        if expect == "accept":
            ok = rc == 0 and summary.get("verdict") == "PASS"
            ev = ("exit=0 verdict=PASS" if ok
                  else f"exit={rc} verdict={summary.get('verdict')}"
                       f" fails={summary.get('failures')}"
                       + (" || " + " || ".join(ev_lines[:3]) if ev_lines else ""))
        else:
            ev_hit = bool(pat and re.search(pat, out)) and \
                summary.get("verdict") == "FAIL"
            ok = rc != 0 and ev_hit
            ev = (f"exit={rc} {ev_lines[0]}" if ev_hit and ev_lines
                  else f"exit={rc} verdict={summary.get('verdict')} "
                       f"expected_evidence_line={'found' if ev_hit else 'MISSING'}"
                       + (" || " + " || ".join(ev_lines[:3]) if ev_lines else ""))
        res = "PASS" if ok else "GAP"
        if not ok:
            gaps.append(case)
        rows.append(f"{case:<22} expected={expect:<7} result={res:<3} {ev}")

    matrix = ("qkv_layout_gate negative-rejection matrix  "
              f"(seed={SEED}, thresholds: cos-min 0.999 relrms-max 0.05 "
              "expect-layout grouped, defaults not loosened)\n"
              + "\n".join(rows) + "\n"
              + ("ALL EXPECTED\n" if not gaps else "GAPS: " + ", ".join(gaps) + "\n"))
    with open(MATRIX, "w") as f:
        f.write(matrix)
    shutil.copyfile(MATRIX, os.path.abspath(EVIDENCE))
    print(matrix)
    print(f"matrix: {MATRIX}")
    print(f"evidence copy: {os.path.abspath(EVIDENCE)}")
    if gaps:
        print(f"GAPS: {gaps}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
