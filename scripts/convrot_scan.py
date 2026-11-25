#!/usr/bin/env python3
"""convrot_scan — audit EVERY tensor of the quantized DiT safetensors against
the finalized convrot semantics (docs/design/2026-11-25_convrot量化語意定案.md).

Serialization facts (measured, convrot_ref.py lineage):
  * semantics sidecar = per-tensor header entry "<tensor>.comfy_quant"
    (JSON bytes), scale = sibling tensor "<...>.weight_scale" (F32 [N])
  * __metadata__ carries only "config"

For each I8 tensor:
  convrot: dequant(q, row_scale, H_gs conference) -> bf16, then requant
    round-trip mirror (rotate bf16, scale=absmax/127 clamp 1e-30, divide
    bf16, RNE, clamp) -> exact / within1 / scale rel / max code dev
  plain int8_tensorwise: rowwise round-trip
Flags: unknown formats, missing sidecar/scale, gs not power of 4, K%gs!=0.

CPU torch (fp32/bf16 matmul). No weight dumps. JSON summary -> OUT.
"""
import json, mmap, os, struct
import torch

MODEL_DIR = os.environ.get("CONVROT_MODEL_DIR",
    "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8")
MODEL_FILE = os.environ.get("CONVROT_MODEL_FILE",
    "minimax_h3_ref2va_int8_convrot.safetensors")
OUT = os.environ.get("CONVROT_OUT",
    "/Users/cyliu99/.hermes/cache/scratch/convrot_sem")

def kron_manual(a, b):
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(
        a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])

HCACHE = {}
def getH(gs):
    if gs not in HCACHE:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                           [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = torch.tensor([[1.0]])
        while h.shape[0] < gs:
            h = kron_manual(h, h4)
        HCACHE[gs] = h / (gs ** 0.5)
    return HCACHE[gs]

path = os.path.join(MODEL_DIR, MODEL_FILE)
with open(path, "rb") as f:
    ln = struct.unpack("<Q", f.read(8))[0]
    hdr_bytes = f.read(ln)
hdr = json.loads(hdr_bytes)
DS = 8 + ln  # data starts right after the 8-byte length field + ln JSON bytes
_f = open(path, "rb")
mm = mmap.mmap(_f.fileno(), 0, access=mmap.ACCESS_READ)

def raw(name):
    info = hdr.get(name)
    if info is None:
        return None
    a, b = info["data_offsets"]
    return mm[DS + a:DS + b]

results, flags = {}, []
sidecar_keys_seen, format_kinds = {}, {}
subjects = [n for n in hdr if n != "__metadata__"
            and not n.endswith(".comfy_quant")
            and not n.endswith(".weight_scale")
            and not n.endswith("_scale")]

for name in subjects:
    info = hdr[name]
    dtype, shape = info["dtype"], info["shape"]
    if dtype != "I8":
        tag = f"{dtype}:-"
        format_kinds[tag] = format_kinds.get(tag, 0) + 1
        continue
    stem = name[: -len(".weight")] if name.endswith(".weight") else None
    sb = (raw(stem + ".comfy_quant") if stem is not None else None) \
         or raw(name + ".comfy_quant")
    if sb is None:
        flags.append(f"{name}: I8 without comfy_quant sidecar")
        continue
    sem = json.loads(sb.decode("utf-8"))
    for k in sem:
        sidecar_keys_seen[k] = sidecar_keys_seen.get(k, 0) + 1
    fmt = sem.get("format")
    tag = f"{dtype}:{fmt}"
    format_kinds[tag] = format_kinds.get(tag, 0) + 1
    n, k = shape[0], shape[1]
    scale_raw = raw(name[:-len(".weight")] + ".weight_scale") if name.endswith(".weight") else None
    if scale_raw is None:
        flags.append(f"{name}: no sibling weight_scale")
        continue
    scales = torch.frombuffer(bytearray(scale_raw[: 4 * n]), dtype=torch.float32).view(n, 1)
    q = torch.frombuffer(bytearray(raw(name)), dtype=torch.int8).view(n, k)
    gs = sem.get("convrot_groupsize")
    if fmt == "int8_tensorwise" and sem.get("convrot"):
        if not gs or gs & (gs - 1) or bin(gs).count("1") != 1 or (gs.bit_length() - 1) % 2:
            flags.append(f"{name}: groupsize {gs} not power of 4")
            continue
        if k % gs:
            flags.append(f"{name}: K={k} not multiple of gs={gs}")
            continue
        H = getH(gs)
        w = ((q.float() * scales).view(n, k // gs, gs) @ H.T).reshape(n, k)
        wb = w.bfloat16()
        wr = (wb.view(n, k // gs, gs) @ getH(gs).bfloat16()).reshape(n, k)
        wr_absmax = wr.abs().float().amax(dim=1, keepdim=True)
        # upstream scale math ORDER: divide first, then clamp (zero rows
        # store exactly 1e-30; pre-clamping then dividing differs 127x and
        # would fake a 0.992 rel err on adaLN-zero-init rows, e.g.
        # blocks.49.adaln_proj last 21504 rows — verified 2026-11-25).
        s2 = (wr_absmax / 127.0).clamp(min=1e-30)
        codes = torch.round(wr.float() / s2).clamp(-128, 127).to(torch.int8)
        d = (codes.float() - q.float())
        exact = float((codes == q).float().mean())
        within1 = float((d.abs() <= 1).float().mean())
        floor = scales.squeeze(1) < 1e-29
        floor_ok = bool((q[floor] == 0).all()) if floor.any() else True
        keep = ~floor
        srel = float(((s2.squeeze(1)[keep] - scales.squeeze(1)[keep]).abs()
                      / scales.squeeze(1)[keep]).max()) if keep.any() else 0.0
        ok = within1 >= 0.9999 and srel <= 1e-2 and floor_ok
        results[name] = dict(kind="convrot", gs=gs, exact=exact, within1=within1,
                             scale_rel=srel, max_code_dev=int(d.abs().max()),
                             zero_scale_rows=int(floor.sum()), floor_ok=floor_ok,
                             verdict="PASS" if ok else "FAIL")
        if not ok:
            flags.append(f"{name}: convrot within1={within1:.6f} srel={srel:.2e}")
    elif fmt == "int8_tensorwise":
        w = (q.float() * scales).bfloat16().float()
        s2 = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
        codes = torch.round(w / s2).clamp(-128, 127).to(torch.int8)
        d = (codes.float() - q.float())
        exact = float((codes == q).float().mean())
        within1 = float((d.abs() <= 1).float().mean())
        floor = scales.squeeze(1) < 1e-29
        floor_ok = bool((q[floor] == 0).all()) if floor.any() else True
        keep = ~floor
        srel = float(((s2.squeeze(1)[keep] - scales.squeeze(1)[keep]).abs()
                      / scales.squeeze(1)[keep]).max()) if keep.any() else 0.0
        ok = within1 >= 0.9999 and srel <= 1e-2 and floor_ok
        results[name] = dict(kind="plain", gs=None, exact=exact, within1=within1,
                             scale_rel=srel, max_code_dev=int(d.abs().max()),
                             zero_scale_rows=int(floor.sum()), floor_ok=floor_ok,
                             verdict="PASS" if ok else "FAIL")
        if not ok:
            flags.append(f"{name}: plain within1={within1:.6f} srel={srel:.2e}")
    else:
        flags.append(f"{name}: I8 format={fmt} NOT COVERED by scan")

conv = [r for r in results.values() if r["kind"] == "convrot"]
fails = [n for n, r in results.items() if r["verdict"] == "FAIL"]
summary = dict(
    file=MODEL_FILE, subjects=len(subjects), audited=len(results),
    convrot_tensors=len(conv),
    format_kinds=format_kinds, sidecar_keys_seen=sidecar_keys_seen,
    convrot_min_exact=min((r["exact"] for r in conv), default=None),
    convrot_min_within1=min((r["within1"] for r in conv), default=None),
    convrot_max_scale_rel=max((r["scale_rel"] for r in conv), default=None),
    convrot_max_code_dev=max((r["max_code_dev"] for r in conv), default=None),
    fails=fails, flags=flags[:80])
os.makedirs(OUT, exist_ok=True)
json.dump(dict(summary=summary, per_tensor=results),
          open(f"{OUT}/convrot_scan.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:3500])
print("OVERALL:", "PASS" if not fails and not any("NOT COVERED" in f or "not power of 4" in f or "not multiple" in f or "without" in f for f in flags) else "FAIL")
