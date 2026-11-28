# Row-order measurement across h3 fused-projection assets. CPU-only, read-only.
import sys, json, gc
import numpy as np
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
from st_scan import (open_st, raw, deq_bf16, deq_int8_scaled, deq_sidecar_scale,
                     hadamard, grouped_from_interleaved, interleave_from_grouped,
                     fused_keys, fp8_e4m3_to_f32)

H256 = hadamard(256)  # orthonormal & symmetric (h4 kron x4), f32
HF_SHARDS = [f"/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/transformer/model-{i:05d}-of-00013.safetensors"
             for i in range(1, 14)]

_NAME2SHARD = {}
def _find_shard(name):
    if not _NAME2SHARD:
        for p in HF_SHARDS:
            st = open_st(p)
            for k in st["hdr"]:
                if not k.endswith(".comfy_quant") and not k.endswith(".weight_scale"):
                    _NAME2SHARD[k] = p
            st["f"].close()
    return _NAME2SHARD[name]

_hf_cache = {}
def hf_tensor(name):
    if name in _hf_cache:
        return _hf_cache[name]
    if len(_hf_cache) > 1:
        _hf_cache.clear(); gc.collect()
    st = open_st(_find_shard(name))
    t = deq_bf16(st, name).astype(np.float32)
    st["f"].close()
    _hf_cache[name] = t
    return t

_hf_ms = {}
def hf_u16_ms(name):
    if name not in _hf_ms:
        _hf_ms[name] = np.sort(np.frombuffer(bytes(raw(open_st(_find_shard(name)), name)), np.uint16))
    return _hf_ms[name]

def asset_u16_ms(st, name):
    return np.sort(np.frombuffer(bytes(raw(st, name)), np.uint16))

def deq_f16(st, name):
    return np.frombuffer(bytes(raw(st, name)), np.uint16).view(np.float16).astype(np.float32).reshape(st["hdr"][name]["shape"])

def dequant_unrot(st, name, scale_key=None):
    codes = deq_int8_scaled(st, name, scale_key or (name[:-7] + ".weight_scale")).astype(np.float32)
    r, c = codes.shape
    return (codes.reshape(r, c // 256, 256) @ H256.T).reshape(r, c)

def cos_sim(a, b):
    a = a.ravel(); b = b.ravel()
    step = 8 * 1024 * 1024
    num = na = nb = 0.0
    for i in range(0, a.size, step):
        ca = a[i:i+step].astype(np.float64); cb = b[i:i+step].astype(np.float64)
        num += float(ca @ cb); na += float(ca @ ca); nb += float(cb @ cb)
    return num / (math.sqrt(na) * math.sqrt(nb)) if na > 0 and nb > 0 else 0.0

import math

def measure(st, name, ref=None, kind="qkv3", quant="bf16", tag="", out=None,
            multiset_vs_hf=False, scale_key=None):
    if quant == "convrot":
        X = dequant_unrot(st, name, scale_key)
    elif quant == "f16":
        X = deq_f16(st, name)
    else:
        X = deq_bf16(st, name).astype(np.float32)
    rec = {"asset": tag, "key": name, "shape": list(st["hdr"][name]["shape"]),
           "dtype": st["hdr"][name]["dtype"], "quant": quant}
    if kind == "qkv3" and X.shape[0] % 384 == 0 and X.ndim == 2:
        rec["c_regroup"] = round(cos_sim(X, grouped_from_interleaved(X)), 5)
        rec["c_ungroup"] = round(cos_sim(X, interleave_from_grouped(X)), 5)
        if ref is not None:
            rec["cos_vs_interleaved"] = round(cos_sim(X, ref), 5)
            rec["cos_vs_grouped"] = round(cos_sim(X, grouped_from_interleaved(ref)), 5)
    else:  # fc1-style 2-kind halves
        h = X.shape[0] // 2
        swapped = np.concatenate([X[h:], X[:h]])
        rec["c_halfswap"] = round(cos_sim(X, swapped), 5)
        if ref is not None:
            rec["cos_vs_identity"] = round(cos_sim(X, ref), 5)
            rh = ref.shape[0] // 2
            rec["cos_vs_halfswap"] = round(cos_sim(X, np.concatenate([ref[rh:], ref[:rh]])), 5)
    if quant in ("bf16", "f16") and multiset_vs_hf and name in _NAME2SHARD or \
       (quant in ("bf16", "f16") and multiset_vs_hf):
        try:
            rec["multiset_match_hf"] = bool(np.array_equal(asset_u16_ms(st, name), hf_u16_ms(name)))
        except KeyError:
            pass
    del X
    if out is not None:
        out.append(rec); gc.collect()
    return rec

def run_family(st, names, tag, quant, kind, out, hf_ref=True, multiset=False, scale_key=None):
    for n in names:
        ref = None
        if hf_ref:
            try:
                ref = hf_tensor(n)
            except KeyError:
                ref = None
        measure(st, n, ref=ref, kind=kind, quant=quant, tag=tag, out=out,
                multiset_vs_hf=multiset, scale_key=scale_key)

def block_names(base, suffix, idxs):
    return [f"{base}.{i}.{suffix}" for i in idxs]
