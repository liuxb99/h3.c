import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
from qkv_layout_gate import WeightSource
from convrot_quant import quant_rotate_f32, dequant_rotate_f32, qkv_grouped_perm
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
subj="blocks.10.mlp.fc2.weight"
N,K=c.ent(subj)[1]["shape"]; gs,_=c.groupsize(subj)
codes=c.codes_i8(subj).reshape(N,K); scale=c.scale_f32(subj).reshape(N,1)
w=r.rows_f32(subj,0,N)
rot=quant_rotate_f32(torch.from_numpy(w)).numpy()  # [N,K//gs,gs]
rotf=rot.reshape(N,K)
absmax_rot=np.abs(rotf).max(1,keepdims=True)
absmax_raw=np.abs(w).max(1,keepdims=True)
print("official scale stats:", scale.min(), scale.max(), scale.mean())
for name,cand in [("absmax_rot/127",absmax_rot/127),
                  ("absmax_rot*sqrt(1/G)/127",absmax_rot/127/np.sqrt(gs)),
                  ("absmax_rot/127/sqrt(G)",absmax_rot/(127*np.sqrt(gs))),
                  ("absmax_raw/127",absmax_raw/127),
                  ("absmax_raw/127/sqrt(G)",absmax_raw/(127*np.sqrt(gs)))]:
    ratio=scale/cand
    print(f"{name:28s} ratio min/med/max = {ratio.min():.6f} {np.median(ratio):.6f} {ratio.max():.6f} std {ratio.std():.3g}")
# also check official codes*scale unrotated row max == absmax_rot? row-wise max of |codes*scale|
mm=(np.abs(codes*scale).max(1,keepdims=True))
print("rowmax|codes*scale| / absmax_rot median:", float(np.median(mm/ (absmax_rot+1e-30))))
print("rowmax|codes*scale| / absmax_raw median:", float(np.median(mm/ (absmax_raw+1e-30))))
