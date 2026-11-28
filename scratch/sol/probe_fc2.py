import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
from convrot_quant import rotate_stages_f32, quant_tensor_f32line
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
subj="blocks.10.mlp.fc2.weight"
gs,cj=c.groupsize(subj); print("comfy_quant:",cj)
N=14336;K=5376
codes=c.codes_i8(subj).reshape(N,K); scale=c.scale_f32(subj)
print("scale stats: min",scale.min(),"max",scale.max(),"zero rows",(scale<=1e-30).sum())
x=codes.astype(np.float32)*scale
w_hf=r.rows_f32(subj,0,N)
def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
d_rot=rotate_stages_f32(torch.from_numpy(x.reshape(N,K//gs,gs))).numpy().reshape(N,K)
print("cos(dequant_rotated, HF)  =", cos(d_rot,w_hf))
print("cos(codes*scale RAW     , HF)  =", cos(x,w_hf))
q,qs=quant_tensor_f32line(torch.from_numpy(w_hf),gs)
q=q.numpy(); qs=qs.numpy()
d_our=rotate_stages_f32((q.astype(np.float32)*qs).reshape(N,K//gs,gs)).numpy().reshape(N,K)
print("cos(our requant dequant, HF) =", cos(d_our,w_hf))
print("codes bitwise equal:", np.array_equal(codes,q))
print("multiset equal:", np.array_equal(np.sort(codes.ravel()+128), np.sort(q.ravel()+128)))
print("frac codes differ:", (codes!=q).mean())
print("scale match frac:", (np.abs(scale[:,0]-qs[:,0])/np.maximum(qs[:,0],1e-30)<1e-3).mean())
# compare official codes multiset vs our requant multiset per-row absmax distributions
print("official absmax/127 vs our scale cos:", cos(scale[:,0].astype(np.float64), qs[:,0].astype(np.float64)))
EOF_MARKER_NOT_USED = None
