#!/usr/bin/env python3
"""rename_built_sidecar_strip.py — convert h3c-built sidecar keys from the
append spelling to the production (official/engine) strip spelling IN PLACE:

  "X.weight.weight_scale"  -> "X.weight_scale"
  "X.weight.comfy_quant"   -> "X.comfy_quant"

Safeguards:
- only the header JSON region is rewritten; the new JSON is space-padded to
  the exact old length, so u64 prefix, data start offset, and ALL payload
  bytes are byte-identical (tensor SHAs and payload offsets stay valid).
- old header saved to evidence/convrot_audit_20261128/built_header_before.json
- re-opens after patch and asserts: key set transformed 250+250, payload
  offsets unchanged, no key left with ".weight.weight_scale".
"""
import mmap, json, struct, os, shutil, sys

PATH = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
BAK = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/convrot_audit_20261128/built_header_before.json"

f = open(PATH, "r+b")
mm = mmap.mmap(f.fileno(), 0)
n = struct.unpack("<Q", mm[:8])[0]
raw = bytes(mm[8:8 + n])
h = json.loads(raw)
if not os.path.exists(BAK):
    open(BAK, "wb").write(raw)

new = {}
ren = 0
for k, v in h.items():
    if k.endswith(".weight.weight_scale"):
        k2 = k[: -len(".weight_scale")] + "_scale"
        ren += 1
    elif k.endswith(".weight.comfy_quant"):
        k2 = k[: -len(".comfy_quant") - len(".weight")] + ".comfy_quant"
        ren += 1
    else:
        k2 = k
    new[k2] = v

new_raw = json.dumps(new, separators=(",", ":")).encode()
# pad to exact old length (JSON readers tolerate trailing whitespace)
pad = len(raw) - len(new_raw)
assert pad >= 0, f"new header longer than old ({len(new_raw)} > {len(raw)})"
new_raw += b" " * pad

ds = 8 + n
mm[8:8 + n] = new_raw
mm.flush()
mm.close()
f.close()
print(f"renamed {ren} sidecar keys; header {n} bytes (pad {pad}); data start unchanged")

# verify reopen
f = open(PATH, "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
n2 = struct.unpack("<Q", mm[:8])[0]
h2 = json.loads(mm[8:8 + n2])
mm.close(); f.close()
assert n2 == n, "u64 changed"
assert all(not k.endswith(".weight.weight_scale") for k in h2)
assert all(not k.endswith(".weight.comfy_quant") for k in h2)
ns = sum(1 for k in h2 if k.endswith("_scale") and not k.endswith(".weight_scale") is False)
scales = [k for k in h2 if k.endswith(".weight_scale")]
comfies = [k for k in h2 if k.endswith(".comfy_quant")]
assert len(scales) == 250 and len(comfies) == 250, (len(scales), len(comfies))
# offsets identical to backup header
old = json.loads(open(BAK, "rb").read())
for k2, v in new.items():
    if k2 == "__metadata__":
        continue
    k_old = k2.replace(".weight_scale", ".weight.weight_scale") if k2.endswith(".weight_scale") else (
        k2[: -len(".comfy_quant")] + ".weight.comfy_quant" if k2.endswith(".comfy_quant") else k2)
    assert old[k_old]["data_offsets"] == v["data_offsets"], k2
print("VERIFY OK: 250 scale + 250 comfy strip-spelling keys, offsets identical")
