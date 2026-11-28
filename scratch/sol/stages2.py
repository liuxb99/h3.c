# Stages L1/AD/TE/SUL/VP8/PIKA: cross-asset row-order probes (CPU-only, read-only)
import sys, json, gc
JD = lambda o: json.dumps(o, default=lambda x: float(x) if hasattr(x, "item") else str(x))
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import numpy as np
from measure_lib import (open_st, hf_tensor, dequant_unrot, cos_sim, deq_f16,
                         asset_u16_ms, hf_u16_ms, H256)
from st_scan import (deq_bf16, deq_sidecar_scale, fp8_e4m3_to_f32, raw,
                     grouped_from_interleaved, interleave_from_grouped)

STAGE = sys.argv[1]
OUT = open("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results2_%s.jsonl" % STAGE, "w")

def unit_rows(M):
    A = M.reshape(M.shape[0] // 128, -1)
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-20)

def slab_match(X, R):
    """mean best-slab cos + fraction of argmax==identity between X rows and R rows."""
    ux, ur = unit_rows(X), unit_rows(R)
    C = ux @ ur.T
    n = ux.shape[0]
    best = C.argmax(1)
    return float(C.max(1).mean()), float((best == np.arange(n)).mean())

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)

def nvfp4_deq(st, name):
    packed = np.frombuffer(bytes(raw(st, name)), np.uint8).reshape(st["hdr"][name]["shape"])
    lo = E2M1[packed & 7] * np.where(packed & 8, -1.0, 1.0)
    hi = E2M1[(packed >> 4) & 7] * np.where(packed & 0x80, -1.0, 1.0)
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
    codes[:, 0::2] = lo; codes[:, 1::2] = hi
    scb = np.frombuffer(bytes(raw(st, name[:-7] + ".weight_scale")), np.uint8)
    sc = fp8_e4m3_to_f32(scb).reshape(st["hdr"][name[:-7] + ".weight_scale"]["shape"])
    X = codes * np.repeat(sc, codes.shape[1] // sc.shape[1], axis=1)
    s2k = name[:-7] + ".weight_scale_2"
    if s2k in st["hdr"]:
        s2 = float(np.frombuffer(bytes(raw(st, s2k)), np.float32).reshape(-1)[0])
        X *= s2
    return X

def load_X(st, name, quant):
    dt = st["hdr"][name]["dtype"]
    if quant == "auto":
        quant = {"I8": "convrot", "F8_E4M3": "fp8", "F16": "f16", "BF16": "bf16", "U8": "nvfp4"}[dt]
    if quant == "convrot":
        return dequant_unrot(st, name), quant
    if quant == "nvfp4":
        return nvfp4_deq(st, name), quant
    if quant == "f16":
        return deq_f16(st, name), quant
    if quant == "fp8":
        shape = st["hdr"][name]["shape"]
        v = fp8_e4m3_to_f32(np.frombuffer(bytes(raw(st, name)), np.uint8)).reshape(shape)
        skey = name[:-7] + ".weight_scale"
        e = st["hdr"][skey]
        if len(e["shape"]) == 0:
            o = e["data_offsets"]
            st["f"].seek(o[0])
            s = float(np.frombuffer(st["f"].read(o[1] - o[0]), np.float32)[0])
            return v * s, quant
        sc = deq_sidecar_scale(st, skey)
        return v * np.repeat(sc, shape[1] // sc.shape[1], axis=1), quant
    return deq_bf16(st, name).astype(np.float32), quant

def probe(tag, path, name_file, ref, kind="qkv3", quant="bf16", extra=None, ref_tag=None):
    st = open_st(path)
    if name_file not in st["hdr"]:
        OUT.write(JD({"stage": STAGE, "asset": tag, "key": name_file,
                              "skip": "missing_key"}) + "\n")
        OUT.flush(); st["f"].close(); return
    X, quant = load_X(st, name_file, quant)
    rec = {"stage": STAGE, "asset": tag, "key": name_file, "quant": quant,
           "shape": list(X.shape)}
    if kind == "qkv3":
        cands = [("vs_I_identity", ref), ("vs_G", grouped_from_interleaved(ref))]
    elif kind in ("fc1", "fc1_loraB"):
        h = ref.shape[0] // 2
        cands = [("vs_identity", ref), ("vs_halfswap", np.concatenate([ref[h:], ref[:h]]))]
    elif kind == "self_layout":  # X vs own regroup/interleave (bijection fraction)
        cands = [("X_vs_ownG", grouped_from_interleaved(X)), ("X_vs_ownI", interleave_from_grouped(X))]
        rec["kind"] = "self_layout"
    for label, R in cands:
        mc, idf = slab_match(X, R)
        rec[label + "_slabmean"] = round(mc, 4)
        rec[label + "_idfrac"] = round(idf, 4)
        rec[label + "_cos"] = round(cos_sim(X, R), 5)
    if quant in ("bf16", "f16", "fp8") and kind != "self_layout" and ref_tag is not None:
        try:
            rec["multiset_match_ref"] = ref_tag + ":" + (
                "1" if np.array_equal(asset_u16_ms(st, name_file), hf_u16_ms(ref_name_of(name_file))) else "0")
        except Exception as e:
            rec["multiset_match_ref"] = "err:" + str(e)[:80]
    if extra:
        rec.update(extra(X) if callable(extra) else extra)
    OUT.write(JD(rec) + "\n"); OUT.flush()
    del X, st
    gc.collect()

def ref_name_of(n):
    return n.replace("model.diffusion_model.", "")

IDX = [0, 25, 49]
OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
SINGULARITY = "/Volumes/SSD2T/models/Pika h3 flex加速模型/models/diffusion_models/Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors"
VP_FP8 = "/Volumes/SSD2T/models/VPipe-comfy_h3/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
PRUNED_FL = "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PRUNED_REF = "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FUSED = "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_fused_flex_lora_v1.safetensors"
RT = "/Users/cyliu99/ComfyUI-Shared/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_requant_roundtrip.safetensors"
COMFYORG_REF = "/Volumes/SSD2T/models/Comfy-Org-MiniMax-H3-bf16/diffusion_models/minimax_h3_ref2va_bf16.safetensors"
COMFYORG_FL = "/Volumes/SSD2T/models/Comfy-Org-MiniMax-H3-bf16/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
AD2 = "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors"
AD4 = "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_4step_dwd_adapter_int8_convrot.safetensors"
AD_BASE = "/Users/cyliu99/Downloads/ai_zyefer/pi_h3_2step_dwd_adapter.safetensors"
TE_NVFP4 = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TE_ORIG_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/text_encoder"
SUL = "/Volumes/SSD2T/models/SulphurAI-Sulphur-3/s3_w4a8.safetensors"

def hf_ref(n):
    return hf_tensor(ref_name_of(n))

if STAGE == "L1":
    import os
    org = COMFYORG_REF if os.path.isfile(COMFYORG_REF) else COMFYORG_FL
    jobs = [
        ("OFFICIAL_CONVROT_REF2VA", OFFICIAL, "convrot", ""),
        ("COMFYORG_BF16", org, "bf16", ""),
        ("PRUNED_FL2VA", PRUNED_FL, "convrot", ""),
        ("PRUNED_REF2VA", PRUNED_REF, "convrot", ""),
        ("FUSED_FLEXLORA", FUSED, "convrot", ""),
        ("REQUANT_RT", RT, "convrot", ""),
        ("SINGULARITY_INT8", SINGULARITY, "convrot", "model.diffusion_model."),
        ("VPIPE_FP8_SCALED", VP_FP8, "fp8", ""),
    ]
    for tag, path, quant, pfx in jobs:
        for i in IDX:
            n = pfx + f"blocks.{i}.attn.qkv_proj.weight"
            probe(tag, path, n, hf_ref(n), "qkv3", quant, ref_tag="FL2VA_HF")
            if tag not in ("VPIPE_FP8_SCALED",):  # fc1 too (half-swap class)
                probe(tag, path, pfx + f"blocks.{i}.mlp.fc1.weight",
                      hf_ref(pfx + f"blocks.{i}.mlp.fc1.weight"), "fc1", quant, ref_tag="FL2VA_HF")
        for i in (0, 1):
            n = pfx + f"token_refiner.blocks.{i}.attn.qkv_proj.weight"
            probe(tag, path, n, hf_ref(n), "qkv3", "auto", ref_tag="FL2VA_HF")
            probe(tag, path, pfx + f"token_refiner.blocks.{i}.mlp.fc1.weight",
                  hf_ref(pfx + f"token_refiner.blocks.{i}.mlp.fc1.weight"), "fc1", "auto", ref_tag="FL2VA_HF")

elif STAGE == "AD":
    # adapter cross-probes: quantized convrot adapters vs bf16 base adapter
    base_st = open_st(AD_BASE)
    base_keys = [k for k in base_st["hdr"] if "to_qkv" in k or "to_out" in k]
    rec0 = {"stage": "AD", "base_keys": sorted(base_keys)[:20]}
    OUT.write(JD(rec0) + "\n")
    for bi in (0, 7):
        kn = f"dwd_pixel_refiner.blocks.{bi}.attn.to_qkv.weight"
        if kn in base_st["hdr"]:
            R = deq_bf16(base_st, kn).astype(np.float32) if base_st["hdr"][kn]["dtype"] == "BF16" else deq_f16(base_st, kn)
            for tag, path in (("AD2_CONVROT", AD2), ("AD4_CONVROT", AD4)):
                def mk(R=R):
                    def f(X):
                        mc, idf = slab_match(X, R)
                        return {"vs_base_slabmean": round(mc, 4), "vs_base_idfrac": round(idf, 4),
                                "vs_base_cos": round(cos_sim(X, R), 5)}
                    return f
                probe(tag, path, kn, R, "self_layout", "convrot", extra=mk)
            # layout thirds evidence: kind-partition norms under each hypothesis
            for tag, path in (("AD2_CONVROT", AD2), ("AD4_CONVROT", AD4)):
                X = load_X(open_st(path), kn, "convrot")
                th = [float(np.linalg.norm(X[r*2048:(r+1)*2048])) for r in range(3)]
                blk = [float(np.linalg.norm(X[i*384:(i+1)*384])) for i in range(16)]
                OUT.write(JD({"stage": "AD", "asset": tag, "key": kn,
                                      "third_norms": [round(t, 1) for t in th],
                                      "block384_norms_cv": round(float(np.std(blk) / np.mean(blk)), 3)}) + "\n")
                del X; gc.collect()
    # fc1 lora_B cross: convrot adapters pass through loras as F16; compare vs base
    for bi in (0, 25):
        kn = f"blocks.{bi}.mlp.fc1.lora_B.weight"
        if kn in base_st["hdr"]:
            R = deq_f16(base_st, kn)
            for tag, path in (("AD2_CONVROT", AD2), ("AD4_CONVROT", AD4)):
                try:
                    probe(tag, path, kn, R, "fc1_loraB", "f16",
                          extra=lambda X: {"note": "X vs base B slab idfrac",
                                           **dict(zip(["idfrac"], [round(slab_match(X, R)[1], 4)]))})
                except KeyError:
                    pass
    base_st["f"].close()

elif STAGE == "TE":
    import glob, re
    shards = sorted(glob.glob(TE_ORIG_DIR + "/model-*.safetensors"))
    orig = {}
    for p in shards:
        st = open_st(p)
        for k in list(st["hdr"]):
            if k.startswith("model.visual.") and ("qkv" in k or "linear_fc1" in k):
                orig[k] = p
        st["f"].close()
    def te_orig(name):
        p = orig["model." + name]
        st = open_st(p)
        t = deq_bf16(st, "model." + name).astype(np.float32)
        st["f"].close()
        return t
    stn = open_st(TE_NVFP4)
    for i in (0, 13, 26):
        n = f"visual.blocks.{i}.attn.qkv.weight"
        probe("TE_NVFP4_visual", TE_NVFP4, n, te_orig(n), "qkv3", "auto", ref_tag="TE_ORIG")
    stn["f"].close()
    stn = open_st(TE_NVFP4)
    for n, tag in [("visual.merger.linear_fc1.weight", "fc1")]:
        probe("TE_NVFP4_visual", TE_NVFP4, n, te_orig(n), "fc1", "auto")
    stn["f"].close()
    # TE int8-convrot variant (VPipe-comfy_h3/text_encoders): visual qkv + merger fc1
    TE_I8 = "/Volumes/SSD2T/models/VPipe-comfy_h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    sti = open_st(TE_I8)
    pref = "model." if any(k.startswith("model.visual.") for k in sti["hdr"]) else ""
    sti["f"].close()
    try:
        for i in (0, 13, 26):
            n = pref + f"visual.blocks.{i}.attn.qkv.weight"
            probe("TE_INT8CONVROT", TE_I8, n, te_orig(n[len(pref):]), "qkv3", "convrot", ref_tag="TE_ORIG")
        for n in [pref + "visual.merger.linear_fc1.weight"]:
            probe("TE_INT8CONVROT", TE_I8, n, te_orig(n[len(pref):]), "fc1", "convrot")
    except (ValueError, KeyError) as exc:
        OUT.write(JD({"stage": "TE_INT8CONVROT", "note": "dequant layout unresolved: " + str(exc)[:120],
             "qkv_hdr_dtype_shape": [sti["hdr"].get(pref + "visual.blocks.0.attn.qkv.weight" + s, {}).get("dtype")
                                     for s in ("", ".weight_scale")]}) + "\n"); OUT.flush()
    # nvfp4 lm q_proj row preservation vs TE orig (not fused; row-order only)
    stn = open_st(TE_NVFP4)
    qn = "model.language_model.layers.0.self_attn.q_proj.weight"
    print("nvfp4 scale dtype/shape:", stn["hdr"].get(qn + ".weight_scale", {}).get("dtype"),
          stn["hdr"].get(qn + ".weight_scale", {}).get("shape"))
    X = nvfp4_deq(stn, qn)
    stn["f"].close()
    R = None
    for p in shards:
        st = open_st(p)
        if qn in st["hdr"]:
            R = deq_bf16(st, qn).astype(np.float32)
        st["f"].close()
        if R is not None:
            break
    mc, idf = slab_match(X, R)
    OUT.write(JD({"stage": "TE", "asset": "TE_NVFP4_lm_q_proj", "key": qn,
                          "vs_orig_slabmean": round(mc, 4), "vs_orig_idfrac": round(idf, 4),
                          "vs_orig_cos": round(cos_sim(X, R), 5), "fused": False}) + "\n")
    del X, R
    gc.collect()

elif STAGE == "SUL":
    st = open_st(SUL)
    def sul_deq(name):
        packed = np.frombuffer(bytes(raw(st, name)), np.uint8).reshape(st["hdr"][name]["shape"])
        cb = np.frombuffer(bytes(raw(st, name + "_codebook")), np.float16).astype(np.float32)
        if cb.shape[0] != 16:
            cb = cb.reshape(1, -1)[:, :16]
        sc = np.frombuffer(bytes(raw(st, name + "_s_channel")), np.float16).astype(np.float32)
        lo = cb[packed & 15]; hi = cb[(packed >> 4) & 15]
        codes = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
        codes[:, 0::2] = lo; codes[:, 1::2] = hi
        assert codes.shape[1] == sc.shape[0], (codes.shape, sc.shape)
        return codes * sc[None, :]
    for i in IDX:
        n = f"blocks.{i}.attn.qkv_proj.weight"
        X = sul_deq(n)
        ref = hf_ref(n)
        for label, R in (("vs_I", ref), ("vs_G", grouped_from_interleaved(ref))):
            mc, idf = slab_match(X, R)
            OUT.write(JD({"stage": "SUL", "asset": "SULPHUR3", "key": n, "cand": label,
                                  "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del X, ref; gc.collect()
        n2 = f"blocks.{i}.mlp.fc1.weight"
        X = sul_deq(n2); ref = hf_ref(n2)
        h = ref.shape[0] // 2
        for label, R in (("vs_identity", ref), ("vs_halfswap", np.concatenate([ref[h:], ref[:h]]))):
            mc, idf = slab_match(X, R)
            OUT.write(JD({"stage": "SUL", "asset": "SULPHUR3", "key": n2, "cand": label,
                                  "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del X, ref; gc.collect()
    for i in (0, 1):
        n = f"token_refiner.blocks.{i}.attn.qkv_proj.weight"
        X = sul_deq(n); ref = hf_ref(n)
        for label, R in (("vs_I", ref), ("vs_G", grouped_from_interleaved(ref))):
            mc, idf = slab_match(X, R)
            OUT.write(JD({"stage": "SUL", "asset": "SULPHUR3", "key": n, "cand": label,
                                  "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del X, ref; gc.collect()
    st["f"].close()

elif STAGE == "VP8":
    import glob
    st = open_st(SUL)
    st["f"].close()
    shards = sorted(glob.glob("/Volumes/SSD2T/models/VPipe-FL2VA-8bit/diffusion_models/model-*.safetensors"))
    name2shard = {}
    for p in shards:
        s = open_st(p)
        for k in s["hdr"]:
            if k.endswith("attn.qkv_proj.weight") or k.endswith("mlp.fc1.weight"):
                name2shard[k] = p
        s["f"].close()
    def vp8_deq(name):
        p = name2shard[name]
        s = open_st(p)
        u = np.frombuffer(bytes(raw(s, name)), np.uint32).reshape(s["hdr"][name]["shape"])
        sc = deq_sidecar_scale(s, name[:-7] + ".scales")
        bi = deq_sidecar_scale(s, name[:-7] + ".biases")
        s["f"].close()
        K = u.shape[1] * 8  # 8 nibbles per u32
        codes = np.empty((u.shape[0], K), np.uint8)
        for sh in range(8):
            codes[:, sh::8] = ((u >> (4 * sh)) & 15).astype(np.uint8)
        vals = (codes.astype(np.float32) - 8.0)
        vals = vals * np.repeat(sc, K // sc.shape[1], axis=1)
        vals = vals + np.repeat(bi, K // bi.shape[1], axis=1)
        return vals
    for i in IDX:
        n = f"blocks.{i}.attn.qkv_proj.weight"
        X = vp8_deq(n); ref = hf_ref(n)
        for label, R in (("vs_I", ref), ("vs_G", grouped_from_interleaved(ref))):
            mc, idf = slab_match(X, R)
            OUT.write(JD({"stage": "VP8", "asset": "VPIPE8_4bit", "key": n, "cand": label,
                                  "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del X, ref; gc.collect()
    for i in (0, 1):
        n = f"token_refiner.blocks.{i}.attn.qkv_proj.weight"
        if n not in name2shard:
            continue
        X = vp8_deq(n); ref = hf_ref(n)
        for label, R in (("vs_I", ref), ("vs_G", grouped_from_interleaved(ref))):
            mc, idf = slab_match(X, R)
            OUT.write(JD({"stage": "VP8", "asset": "VPIPE8_4bit", "key": n, "cand": label,
                                  "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del X, ref; gc.collect()

elif STAGE == "PIKA":
    P = "/Volumes/SSD2T/models/Pika h3 flex加速模型/models/loras/"
    import os
    cands = [p for p in ["pika_h3_flex_fl2va_v1_bf16.safetensors",
                         "pika_h3_flex_ref2va_v1_bf16.safetensors",
                         "pika_h3_flex_ref2va_comfy_bf16.safetensors",
                         "pika_h3_flex_turbo_8step_hd_ref2va.safetensors"]
             if os.path.isfile(P + p)]
    if not cands:
        import glob
        g = glob.glob("/Volumes/SSD2T/models/Pika h3 flex加速模型/models/**/pika_h3_flex*bf16.safetensors", recursive=True)
        cands = [os.path.abspath(x) for x in g]
    def getB(p, key):
        s = open_st(p if os.path.isabs(p) else P + p)
        B = deq_bf16(s, key).astype(np.float32)
        s["f"].close()
        return B
    for key in ("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
                "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight"):
        Bs = {}
        for p in cands:
            try:
                Bs[p] = getB(p, key)
            except Exception as e:
                OUT.write(JD({"stage": "PIKA", "file": p, "key": key, "error": str(e)[:120]}) + "\n")
        names = list(Bs)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                X, R = Bs[names[a]], Bs[names[b]]
                mc, idf = slab_match(X, R)
                mcg, idfg = slab_match(X, grouped_from_interleaved(R))
                OUT.write(JD({"stage": "PIKA", "key": key, "pair": (names[a], names[b]),
                                      "vs_identity_idfrac": round(idf, 4), "vs_identity_slabmean": round(mc, 4),
                                      "vs_grouped_idfrac": round(idfg, 4), "vs_grouped_slabmean": round(mcg, 4)}) + "\n")
        del Bs; gc.collect()
else:
    raise SystemExit("no stage " + STAGE)

OUT.close()
print("DONE", STAGE)
