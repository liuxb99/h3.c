import sys; sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
import numpy as np
from measure_lib import open_st, hf_tensor, dequant_unrot
from st_scan import grouped_from_interleaved, deq_int8_scaled

OFF = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
N = "blocks.0.attn.qkv_proj.weight"
hf = hf_tensor(N)                       # interleaved I (56 heads x 384 rows)
st = open_st(OFF)
X = dequant_unrot(st, N)                # official un-rotated rows, file order
codes = deq_int8_scaled(st, N, N[:-7] + ".weight_scale").astype(np.float32)  # rotated-space rows
st["f"].close()

def slabs(M):
    return M.reshape(168, 128, M.shape[1])
def unit_cos(A, B):
    A = A.reshape(168, -1); B = B.reshape(168, -1)
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T

Xs, Hs = slabs(X), slabs(hf)
C = unit_cos(Xs, Hs)
best = C.argmax(1); bestc = C.max(1)
print("X vs I: best-cos min %.4f mean %.4f  identity? %s  unique-argmax %d/168"
      % (bestc.min(), bestc.mean(), (best == np.arange(168)).all(), len(set(best.tolist()))))
HG = slabs(grouped_from_interleaved(hf))
Cg = unit_cos(Xs, HG)
bestg = Cg.argmax(1); bestgc = Cg.max(1)
print("X vs G: best-cos min %.4f mean %.4f  identity? %s  unique %d/168"
      % (bestgc.min(), bestgc.mean(), (bestg == np.arange(168)).all(), len(set(bestg.tolist()))))
Cr = unit_cos(slabs(codes), Hs)
print("codes(rotated space) vs I: mean best-cos %.4f max %.4f" % (Cr.max(1).mean(), Cr.max(1).max()))
print("first 12 argmax(X->I):", best[:12].tolist())
print("first 12 argmax(X->G):", bestg[:12].tolist())
np.save("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/slab_best_I.npy", best)
