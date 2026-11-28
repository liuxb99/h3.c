import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
from qkv_layout_gate import WeightSource
from convrot_quant import h_build
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
h4=np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
def kronrep(G,M):
    K=np.array([[1.0]],dtype=np.float32)
    while K.shape[0]<G: K=np.kron(K,M)
    return K
def stages_msb(x,G):
    a=np.ascontiguousarray(x.reshape(-1,G)); M=a.shape[0]; half=1
    while half<G:
        full=half*4; b=a.reshape(M,half,4,G//full)
        b0,b1,b2,b3=[b[:,:,k].copy() for k in range(4)]
        b[:,:,0]=b0+b1+b2-b3; b[:,:,1]=b0+b1-b2+b3; b[:,:,2]=b0-b1+b2+b3; b[:,:,3]=-b0+b1+b2+b3
        a=b.reshape(M,G); half=full
    return a.reshape(x.shape)
subj="blocks.10.mlp.fc2.weight"
N,K=c.ent(subj)[1]["shape"]; gs,_=c.groupsize(subj)
codes=c.codes_i8(subj).reshape(N,K); scale=c.scale_f32(subj).reshape(N,1)
w=r.rows_f32(subj,0,N); R=w.reshape(N,K//gs,gs); S=scale.reshape(N,1,1)
Hb=h_build(gs); Ks=kronrep(gs,h4)
cands={
 "v@Hb": (codes.reshape(N,K//gs,gs)*S)@Hb,
 "v@Hbᵀ": (codes.reshape(N,K//gs,gs)*S)@Hb.T,
 "v@Ks": (codes.reshape(N,K//gs,gs)*S)@Ks,
 "v@Ksᵀ": (codes.reshape(N,K//gs,gs)*S)@Ks.T,
 "stages_msb(v)": stages_msb((codes.reshape(N,K//gs,gs)*S),gs).reshape(N,K),
 "stages_lsb(v)": (lambda a: a)(None) if False else None,
}
del cands["stages_lsb(v)"]
# also quant-side check: rotate W forward with each, requant, compare codes
import torch as T
for name,X in cands.items():
    dq=np.asarray(X).reshape(N,K)
    print(f"{name:16s} cos(dequant,W)={cos(dq,w):.6f}")
for name,Rq in {"W@Ks":(R@Ks),"W@Ksᵀ":(R@Ks.T),"W@Hb":(R@Hb),"W@Hbᵀ":(R@Hb.T),"stages_msb(W)":stages_msb(R,gs)}.items():
    Rq=np.asarray(Rq,dtype=np.float32); rot=Rq.reshape(N,K)
    am=np.abs(rot).max(1,keepdims=True); sc=am/127
    q=T.from_numpy(rot.reshape(N,K)/sc).round_().clamp_(-127,127).numpy().astype(np.int8)
    print(f"quant {name:14s} codes match official: {float((q==codes).mean()):.7f}  scale ratio med: {float(np.median(scale/(sc+1e-30))):.4f}")
