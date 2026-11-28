import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
from convrot_quant import dequant_f32, quant_tensor_f32line
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
for subj in ["blocks.10.mlp.fc2.weight","blocks.10.mlp.fc1.weight","blocks.10.attn.out_proj.weight"]:
    gs,_=c.groupsize(subj)
    shape=c.ent(subj)[1]["shape"]
    codes=c.codes_i8(subj); scale=c.scale_f32(subj)
    d=dequant_f32(torch.from_numpy(codes.copy()),torch.from_numpy(scale.copy()),gs).numpy()
    w=r.rows_f32(subj,0,shape[0])
    print(subj, shape, "gs",gs)
    print("  cos(dequant, HF)  =", cos(d,w))
    q,qs=quant_tensor_f32line(torch.from_numpy(w),gs)
    print("  codes bitwise == official:", np.array_equal(codes, q.numpy()))
    print("  scale bitwise == official:", np.array_equal(scale, qs.numpy().reshape(scale.shape)))
