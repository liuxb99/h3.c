import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
from convrot_quant import quant_tensor_f32line, dequant_f32, qkv_grouped_perm
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
for subj in ["blocks.10.mlp.fc2.weight","blocks.10.mlp.fc1.weight","blocks.10.attn.out_proj.weight",
             "blocks.10.attn.qkv_proj.weight","token_refiner.blocks.0.attn.qkv_proj.weight",
             "blocks.0.adaln_proj.linear.weight"]:
    gs,_=c.groupsize(subj)
    N,K=c.ent(subj)[1]["shape"]
    codes=c.codes_i8(subj).reshape(N,K); scale=c.scale_f32(subj).reshape(N,1)
    w=r.rows_f32(subj,0,N)
    q,qs=quant_tensor_f32line(torch.from_numpy(w),gs)
    q=q.numpy(); qs=qs.numpy().reshape(scale.shape)
    # candidate comparison domains: file order (codes) vs requant(HF rows) and requant(grouped)
    import numpy as _np
    if subj.endswith("attn.qkv_proj.weight"):
        qg=qkv_grouped_perm(q); qsg=qkv_grouped_perm(qs)
    else:
        qg, qsg = q, qs
    d=dequant_f32(torch.from_numpy(codes.copy()),torch.from_numpy(scale.copy()),gs).numpy()
    print(subj, c.ent(subj)[1]["shape"], "gs",gs)
    print("  codes match file-vs-requant-inter  :", float((codes==q).mean()))
    print("  codes match file-vs-requant-grouped:", float((codes==qg).mean()))
    print("  scale match grouped   :", float((scale==qsg).mean()))
    # cos dequant(file) vs HF rows: file order vs HF order — for qkv use grouped-permuted HF? dequant gives file-order rows; HF is interleaved.
    print("  cos(dequant_file, HF_inter) :", cos(d, w))
    if subj.endswith("attn.qkv_proj.weight"):
        print("  cos(dequant_file, HF_grouped):", cos(d, qkv_grouped_perm(w)))
