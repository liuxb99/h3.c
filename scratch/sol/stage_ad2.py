# stage_ad2: convrot DWD-adapter row-order signatures (CPU-only, read-only)
import sys, json
JD = lambda o: json.dumps(o, default=lambda x: float(x) if hasattr(x, "item") else str(x))
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import numpy as np
from measure_lib import open_st, dequant_unrot, deq_f16

OUT = open("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results2_AD2.jsonl", "w")
AD2 = "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_2step_dwd_adapter_int8_convrot.safetensors"
AD4 = "/Users/cyliu99/ComfyUI-Shared/models/loras/pi-H3/pi_h3_4step_dwd_adapter_int8_convrot.safetensors"

def kind_signature(X, name, tag):
    """Interleaved 384-block vs grouped 2048-third norms; per-128-slab profile stats."""
    rec = {"stage": "AD2", "asset": tag, "key": name}
    thirds = [np.linalg.norm(X[r * (X.shape[0] // 3):(r + 1) * (X.shape[0] // 3)]) for r in range(3)]
    rec["third_norms"] = [round(float(t), 1) for t in thirds]
    rec["third_norm_ratio_maxmin"] = round(max(thirds) / max(min(thirds), 1e-9), 3)
    nb = X.shape[0] // 384
    blk = [np.linalg.norm(X[i * 384:(i + 1) * 384]) for i in range(nb)]
    rec["block384_norms_mean"] = round(float(np.mean(blk)), 1)
    rec["block384_norms_cv"] = round(float(np.std(blk) / np.mean(blk)), 3)
    # within-block 3-slab (128-row) norm spread averaged over blocks:
    sp = []
    for i in range(nb):
        sl = [np.linalg.norm(X[i * 384 + k * 128:i * 384 + (k + 1) * 128]) for k in range(3)]
        sp.append(max(sl) / max(min(sl), 1e-9))
    rec["within384_slab_ratio_mean"] = round(float(np.mean(sp)), 3)
    OUT.write(JD(rec) + "\n"); OUT.flush()

def unit_rows(M):
    A = M.reshape(M.shape[0] // 128, -1)
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-20)

def slab_id(X, R):
    C = unit_rows(X) @ unit_rows(R).T
    n = C.shape[0]
    return float(C.max(1).mean()), float((C.argmax(1) == np.arange(n)).mean())

stA = open_st(AD2); stB = open_st(AD4)
for bi in (0, 7):
    kn = f"dwd_pixel_refiner.blocks.{bi}.attn.to_qkv.weight"
    XA = dequant_unrot(stA, kn)
    kind_signature(XA, kn, "AD2_CONVROT")
    try:
        XB = dequant_unrot(stB, kn)
        mc, idf = slab_id(XA, XB)
        OUT.write(JD({"stage": "AD2", "pair": "AD2vsAD4", "key": kn,
                              "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del XB
    except KeyError:
        pass
    # bias signature (fp16 passthrough rows follow weight rows)
    bkn = f"dwd_pixel_refiner.blocks.{bi}.attn.to_qkv.bias"
    if bkn in stA["hdr"]:
        b = deq_f16(stA, bkn).reshape(-1, 1)
        kind_signature(b, bkn, "AD2_CONVROT_bias")
    del XA
OUT.write(JD({"stage": "AD2", "note": "lora cross 2step vs 4step fc1 lora_B"}) + "\n")
for bi in (0, 25):
    kn = f"blocks.{bi}.mlp.fc1.lora_B.weight"
    if kn in stA["hdr"] and kn in stB["hdr"]:
        XA = deq_f16(stA, kn); XB = deq_f16(stB, kn)
        mc, idf = slab_id(XA, XB)
        OUT.write(JD({"stage": "AD2", "pair": "loraB_2step_vs_4step", "key": kn,
                              "slabmean": round(mc, 4), "idfrac": round(idf, 4)}) + "\n")
        del XA, XB
stA["f"].close(); stB["f"].close()
OUT.close()
print("DONE AD2")
