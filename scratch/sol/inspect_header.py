#!/usr/bin/env python3
"""Header-only inspection: candidate vs HF golden key inventory. Read-only."""
import json, struct, sys, os

CAND = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
GOLDEN_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"

with open(CAND, "rb") as f:
    ln = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(ln))
DS = 8 + ln
print(f"header_len={ln} DS={DS} entries={len(hdr)}")
assert "offset" not in hdr, "unexpected 'offset' key in header"

meta = hdr.pop("__metadata__", None)
print("meta keys:", list(meta.keys()) if meta else None)

quant, scales, sidecars, passthru = {}, {}, {}, {}
for k, v in hdr.items():
    dt = v["dtype"]
    if k.endswith(".comfy_quant"):
        sidecars[k] = v
    elif k.endswith(".weight_scale"):
        scales[k] = v
    elif dt == "I8":
        quant[k] = v
    else:
        passthru[k] = (dt, tuple(v["shape"]))
print(f"quant={len(quant)} scales={len(scales)} sidecars={len(sidecars)} passthru={len(passthru)}")
dtc = {}
for k, (dt, s) in passthru.items():
    dtc[dt] = dtc.get(dt, 0) + 1
print("passthru dtypes:", dtc)

# sidecar contents
gs_map = {}
bad_side = []
for k, v in sorted(sidecars.items()):
    with open(CAND, "rb") as f:
        f.seek(DS + v["data_offsets"][0])
        js = json.loads(f.read(v["data_offsets"][1] - v["data_offsets"][0]))
    stem = k[: -len(".comfy_quant")]
    gs_map[stem] = js.get("convrot_groupsize")
    extra = set(js.keys()) - {"format", "convrot", "convrot_groupsize"}
    if extra or js.get("format") != "int8_tensorwise" or js.get("convrot") is not True:
        bad_side.append((k, js))
print("sidecar anomalies:", bad_side[:5], "count", len(bad_side))
gsv = {}
for s, g in gs_map.items():
    gsv[g] = gsv.get(g, 0) + 1
print("groupsize histogram:", gsv)

# consistency: every quant subject has scale+sidecar with matching shape
for k, v in quant.items():
    assert k in gs_map, f"no sidecar for {k}"
    sk = k + ".weight_scale"
    assert sk in scales, f"no scale for {k}"
    assert scales[sk]["dtype"] == "F32", (sk, scales[sk]["dtype"])
    assert list(scales[sk]["shape"]) == [v["shape"][0], 1] or list(scales[sk]["shape"]) == [v["shape"][0]], (sk, scales[sk]["shape"])
for sk in scales:
    assert sk[: -len(".weight_scale")] in quant, f"orphan scale {sk}"
for sk in sidecars:
    assert sk[: -len(".comfy_quant")] in quant, f"orphan sidecar {sk}"

# golden index
idx = json.load(open(os.path.join(GOLDEN_DIR, "model.safetensors.index.json")))
gweight = idx["weight_map"]
gkeys = set(gweight)
print(f"golden keys={len(gkeys)} shards={len(set(gweight.values()))}")

cand_tensor_keys = set(quant) | set(passthru)  # scale+sidecar are derived extras
missing_in_golden = cand_tensor_keys - gkeys
missing_in_cand = gkeys - cand_tensor_keys
print("cand-not-in-golden:", sorted(missing_in_golden)[:10], len(missing_in_golden))
print("golden-not-in-cand:", sorted(missing_in_cand)[:10], len(missing_in_cand))

# quant subjects: shape match golden, dtype should be BF16 in golden
mism = []
for k, v in quant.items():
    if k not in gkeys:
        mism.append((k, "absent"))
        continue
print("quant subjects absent from golden:", mism[:10], len(mism))

# qkv-like subjects
qkv = {k: (tuple(v["shape"]), gs_map.get(k)) for k, v in quant.items() if "qkv" in k.lower()}
print("qkv quant subjects:", len(qkv))
for k, v in sorted(qkv.items()):
    gdt = None
    print(" ", k, v)
# passthrough key detail sample + shapes vs golden (dtype/shape check requires shard headers; just list first few)
pt = sorted(passthru.items(), key=lambda x: -(x[1][1][0] if x[1][1] else 0))[:8]
for k, v in pt:
    print("PT-big:", k, v)

# scale byte sizes sanity
tot = 0
for k, v in scales.items():
    tot += v["data_offsets"][1] - v["data_offsets"][0]
tot_i8 = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in quant.values())
tot_pt = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in passthru.values())
tot_sc = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in sidecars.values())
print(f"bytes: codes={tot_i8} scales={tot} sidecar={tot_sc} passthru={tot_pt} total={tot_i8+tot+tot_sc+tot_pt} filesize={os.path.getsize(CAND)}")
