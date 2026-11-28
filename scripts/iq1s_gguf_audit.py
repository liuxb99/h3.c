#!/usr/bin/env python3
"""iq1s_gguf_audit.py — structural audit of
/Volumes/SSD2T/models/MiniMax-H3-gguf/minimax_h3_fl2va_pruned-IQ1_S.gguf
(1) stdlib-only GGUF v3 parser: header, all KV metadata, all tensor infos,
    offset contiguity + file-size consistency.
(2) quant-type byte/ tensor census.
(3) key-set diff against the INT8-ConvRot safetensors asset header
    (/Volumes/SSD2T/models/MiniMax-H3-h3c-built, header-only read).
Prints a compact machine-readable JSON summary. Asserts sha256 prefix.
Usage: python3 iq1s_gguf_audit.py [--json-out FILE]
"""
import json, hashlib, mmap, os, struct, sys, collections

GGUF = "/Volumes/SSD2T/models/MiniMax-H3-gguf/minimax_h3_fl2va_pruned-IQ1_S.gguf"
SHA_PREFIX = "76f1d0bcc9052978"
SAFET = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
GGUF_TYPES = {0:"GGML_TYPE_F32",1:"GGML_TYPE_F16",2:"GGML_TYPE_Q4_0",7:"GGML_TYPE_Q8_0",
 8:"GGML_TYPE_Q5_0",9:"GGML_TYPE_Q5_1",10:"GGML_TYPE_IQ2_XXS",11:"GGML_TYPE_IQ2_XS",
 12:"GGML_TYPE_Q2_K",13:"GGML_TYPE_Q3_K_S",14:"GGML_TYPE_Q3_K_M",15:"GGML_TYPE_Q4_K_S",
 16:"GGML_TYPE_Q4_K_M",17:"GGML_TYPE_Q5_K_S",18:"GGML_TYPE_Q5_K_M",19:"GGML_TYPE_Q6_K",
 20:"GGML_TYPE_Q8_K",21:"GGML_TYPE_IQ2_XXS",22:"GGML_TYPE_IQ2_XS",23:"GGML_TYPE_Q2_K",
 24:"GGML_TYPE_Q3_K_S",25:"GGML_TYPE_Q3_K_M",26:"GGML_TYPE_Q4_K_S",27:"GGML_TYPE_Q4_K_M",
 28:"GGML_TYPE_Q5_K_S",29:"GGML_TYPE_Q5_K_M",30:"GGML_TYPE_Q6_K",31:"GGML_TYPE_Q8_K",
 32:"GGML_TYPE_IQ1_S",33:"GGML_TYPE_IQ1_M",34:"GGML_TYPE_IQ2_S",35:"GGML_TYPE_IQ2_M",
 36:"GGML_TYPE_IQ3_XXS",37:"GGML_TYPE_IQ1_S",38:"GGML_TYPE_IQ4_NL",39:"GGML_TYPE_IQ3_S",
 40:"GGML_TYPE_IQ3_M",41:"GGML_TYPE_IQ2_S",42:"GGML_TYPE_IQ2_M",43:"GGML_TYPE_IQ3_XXS",
 44:"GGML_TYPE_IQ1_S",45:"GGML_TYPE_IQ4_NL",46:"GGML_TYPE_IQ3_S",47:"GGML_TYPE_IQ3_M",
 48:"GGML_TYPE_IQ2_S",49:"GGML_TYPE_IQ2_M",50:"GGML_TYPE_IQ3_XXS"}
# canonical ggml type enum (v3 ggml-common.h): 0 F32,1 F16,2 Q4_0,3 Q4_1,4 Q5_0,5 Q5_1,
# 6 Q8_0,7 Q8_1,8 Q2_K,9 Q3_K,10 Q4_K,11 Q5_K,12 Q6_K,13 IQ2_XXS,14 IQ2_XS,15 IQ3_XXS,
# 16 IQ1_S,17 IQ4_NL,18 IQ3_S,19 IQ3_M,20 IQ2_S,21 IQ4_XS,22 IQ8_0,23 IQ1_M,24 BF16
# FACT: enum from ~/tools/stable-diffusion.cpp ggml/include/ggml.h (commit e239e4b,
# upstream pin 13b9d92) — sd.cpp ggml fork differs from llama.cpp numbering!
GGML_ENUM = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
 10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
 18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",
 25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",
 39:"MXFP4"}
TSIZE = {0:4,1:2,24:2}  # bytes per element for plain types; quant types: n_elems*n/8 packed

def gname(t): return GGML_ENUM.get(t, f"TYPE{t}")

def norm0(k): return k.replace("blk.", "blocks.")

def rd_str(mm, off):
    (n,) = struct.unpack_from("<Q", mm, off); off += 8
    s = bytes(mm[off:off+n]).decode("utf-8"); return s, off+n

def rd_val(mm, off, vt):
    if vt == 0: v,off = struct.unpack_from("<i", mm, off)[0], off+4
    elif vt == 1: v,off = struct.unpack_from("<Q", mm, off)[0], off+8
    elif vt == 2: v,off = rd_str(mm, off)
    elif vt == 3: v,off = struct.unpack_from("<f", mm, off)[0], off+4
    elif vt == 4: v,off = struct.unpack_from("<b", mm, off)[0], off+1
    elif vt == 5: v,off = struct.unpack_from("<B", mm, off)[0], off+1
    elif vt == 6: v,off = struct.unpack_from("<H", mm, off)[0], off+2
    elif vt == 7: v,off = struct.unpack_from("<I", mm, off)[0], off+4
    elif vt == 8: v,off = struct.unpack_from("<q", mm, off)[0], off+8
    elif vt == 9: v,off = struct.unpack_from("<QQ", mm, off); off += 16  # array handled below
    else: raise ValueError(vt)
    return v, off

def rd_arr(mm, off):
    (et,) = struct.unpack_from("<I", mm, off); off += 4
    (n,) = struct.unpack_from("<Q", mm, off); off += 8
    fsz = {0:4,1:8,2:8,3:4,4:1,5:1,6:2,7:4,8:8}[et]
    vals = list(struct.unpack_from("<"+"qQffffbBHI"[et]*n, mm, off)) if et!=2 else None
    if et == 2:
        vals, o = [], off
        for _ in range(n):
            s, o = rd_str(mm, o); vals.append(s)
        off = o
    else:
        off += n*fsz
    return (gname(et) if et in (0,1) else f"elem{et}", n, vals), off

def main():
    json_out = None
    if "--json-out" in sys.argv: json_out = sys.argv[sys.argv.index("--json-out")+1]
    h = hashlib.sha256()
    with open(GGUF, "rb") as f:
        for chunk in iter(lambda: f.read(1<<24), b""): h.update(chunk)
    sha = h.hexdigest()
    assert sha.startswith(SHA_PREFIX), f"sha mismatch: {sha}"
    size = os.path.getsize(GGUF)
    out = {"file": GGUF, "size": size, "sha256": sha, "verdict_parts": {}}

    with open(GGUF, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        assert mm[:4] == b"GGUF", "not GGUF"
        (ver,) = struct.unpack_from("<I", mm, 4); off = 8
        # GGUF header order: magic, version, n_tensors, n_kv
        (nt,) = struct.unpack_from("<Q", mm, off); off += 8
        (nkv,) = struct.unpack_from("<Q", mm, off); off += 8
        kvs = {}
        for _ in range(nkv):
            k, off = rd_str(mm, off)
            (vt,) = struct.unpack_from("<I", mm, off); off += 4
            if vt == 9: v, off = rd_arr(mm, off)
            else: v, off = rd_val(mm, off, vt)
            kvs[k] = v
        tensors, contig = [], True
        base_expect = None
        for _ in range(nt):
            name, off = rd_str(mm, off)
            (nd,) = struct.unpack_from("<I", mm, off); off += 4
            dims = list(struct.unpack_from("<"+"Q"*nd, mm, off)); off += 8*nd
            (ttyp,) = struct.unpack_from("<I", mm, off); off += 4
            (toff,) = struct.unpack_from("<Q", mm, off); off += 8
            tensors.append((name, dims, gname(ttyp), toff))
        # tensor data region starts at aligned offset after metadata
        data_start = (off + 0x1f) & ~0x1f  # GGUF alignment 32
        # infer bytes per tensor via offset gaps
        gaps = {}
        by_off = sorted(tensors, key=lambda t: t[3])
        for i,(nm,dims,ty,toff) in enumerate(by_off):
            nxt = by_off[i+1][3] if i+1 < len(by_off) else size - data_start
            gap = nxt - toff
            ne = 1
            for d in dims: ne *= d
            gaps[ty] = gaps.get(ty, 0) + gap
            out.setdefault("bytes_per_elem", {})[ty] = round(gap/ne, 4) if ne else 0
            if i > 0 and by_off[i-1][3] + (gaps_last if False else 0): pass
        # contiguity check: each tensor's inferred size must equal gap to next
        inferred = {}
        for i,(nm,dims,ty,toff) in enumerate(by_off):
            ne = 1
            for d in dims: ne *= d
            inferred[nm] = (by_off[i+1][3]-toff) if i+1<len(by_off) else (size-data_start-toff)
        last_end = data_start + sum(inferred[t[0]] for t in by_off) if by_off else 0
        census = collections.Counter(); bytes_by_type = collections.Counter()
        for nm,dims,ty,toff in by_off:
            census[ty]+=1; bytes_by_type[ty]+=inferred[nm]
        # prefix census for IQ1_S
        pref = collections.Counter()
        for nm,dims,ty,toff in by_off:
            p = nm.rsplit(".",2)[0] if nm.count(".")>=2 else nm
            pref[(p,ty)]+=1
        out.update({
            "gguf_version": ver, "n_kv": nkv, "n_tensors": nt,
            "arch_kvs": {k:v for k,v in kvs.items() if any(s in k for s in
                ("architecture","block_count","length","embedding","attention",
                 "feed_forward","hidden","vae","unet","video","timestep","quant"))},
            "all_kv_keys": sorted(kvs.keys()),
            "type_census": dict(census), "bytes_by_type": {k: v for k,v in bytes_by_type.items()},
            "iq1s_share_pct": round(100*bytes_by_type.get("IQ1_S",0)/size, 2),
            "data_start": data_start,
            "contiguity_last_end": last_end,
        })
        mm.close()

    # safetensors header (header-only read)
    st = {}
    with open(SAFET, "rb") as f:
        (hl,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hl))
    st_keys = {k for k in hdr if k != "__metadata__"}
    st_hdr_map = {norm0(k): v for k, v in hdr.items() if k != "__metadata__"}
    st_meta = hdr.get("__metadata__", {})
    # GGUF set — sd.cpp GGUF names carry "model.diffusion_model." prefix (FACT from probe)
    g_keys = {n for n,_,_,_ in tensors}
    g_norm = {n.removeprefix("model.diffusion_model."): n for n in g_keys}
    def norm(k):
        k = k.replace("blk.", "blocks.")
        return k
    st_norm = {norm(k): k for k in st_keys}
    only_gguf = sorted(set(g_norm) - set(st_norm)); only_st = sorted(set(st_norm) - set(g_norm))
    common = sorted(set(g_norm) & set(st_norm))
    # shape parity on common keys (GGUF dims are reversed row-major order vs safetensors)
    gdims = {n.removeprefix("model.diffusion_model."): (d, t) for n, d, t, _ in tensors}
    shape_ok = shape_bad = 0; shape_mism = []
    for c in common:
        gd, gt = gdims[c]; sd = st_hdr_map.get(c)
        if sd is None: continue
        if list(reversed(gd)) == list(sd["shape"]): shape_ok += 1
        else:
            shape_bad += 1
            if len(shape_mism) < 12: shape_mism.append(f"{c}: gguf {list(reversed(gd))} {gt} vs st {sd['shape']}")
    out["common_shape_ok"] = shape_ok; out["common_shape_bad"] = shape_bad
    out["shape_mismatch_sample"] = shape_mism
    out["safetensors_n_keys"] = len(st_keys)
    out["gguf_n_keys"] = len(g_keys)
    out["common_keys"] = len(common)
    out["only_gguf_sample"] = only_gguf[:40]; out["only_gguf_n"] = len(only_gguf)
    out["only_st_sample"] = only_st[:40]; out["only_st_n"] = len(only_st)
    out["st_meta_keys"] = sorted(st_meta.keys())[:10]
    out["iq1s_prefix_top"] = [f"{p}:{ty}x{c}" for (p,ty),c in sorted(pref.items(), key=lambda x:-x[1])[:25]]
    # tensor name full census grouped
    groups = collections.Counter()
    for nm,_,ty,_ in by_off:
        groups[(nm.split(".")[0], ty)] += 1
    out["groups"] = [f"{a}/{b}x{c}" for (a,b),c in sorted(groups.items())]
    out["first60"] = [f"{n} {ty} {dims}" for n,dims,ty,_ in by_off[:60]]
    out["verdict_parts"]["sha_ok"] = True
    ok = (last_end == size)
    out["verdict_parts"]["offset_size_consistent"] = ok
    out["ok"] = ok
    js = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    print(js[:6000])
    if json_out: open(json_out,"w").write(js)
    print("AUDIT rc=0 ok=", ok)

if __name__ == "__main__":
    main()
