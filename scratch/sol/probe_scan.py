import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")
def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
h4e=np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
print("engine h4^2 ==4I:", np.array_equal(h4e@h4e, 4*np.eye(4,dtype=np.float32)))
def kron_build(G,M0):
    H=np.array([[1.0]],dtype=np.float32)
    while H.shape[0]<G:
        W=H.shape[0]; g=np.empty((W*4,W*4),dtype=np.float32)
        for i in range(W):
            for j in range(W): g[i*4:(i+1)*4,j*4:(j+1)*4]=H[i,j]*M0
        H=g
    return H
def sylv(G):
    H=np.ones((1,1),dtype=np.float32)
    while H.shape[0]<G: H=np.block([[H,H],[H,-H]])
    return H
G=256
K_e=kron_build(G,h4e)
K_s=sylv(G)
cands={"kron4":K_e,"kron4T":K_e.T,"sylv":K_s,"sylvT":K_s.T}
subj="blocks.10.mlp.fc2.weight"
gs,_=c.groupsize(subj)
N,K=c.ent(subj)[1]["shape"]; print("shape",c.ent(subj)[1]["shape"])
codes=c.codes_i8(subj).astype(np.float32).reshape(N,K//gs,gs); scale=c.scale_f32(subj)
w=r.rows_f32(subj,0,N).reshape(N,K//gs,gs)
# dequant candidate kernel computes codes*scale @ Xᵀ (X=had built from h4 => kernel matrix K_eᵀ); scan X per candidate? kernel X is FIXED = K_eᵀ. So scan the IMPLICIT quant rotation: HF @ Qr == codes*scale (up to rounding)
wflat=w.reshape(N,K); codesf=codes*scale.reshape(N,K)
# requant candidates: rotate HF by R, row-absmax scale, round; compare codes with official codes
import torch as T
for name,R in cands.items():
    if not np.allclose(R@R.T, G*np.eye(G,dtype=np.float32)) and not np.allclose(R@R, G*np.eye(G,dtype=np.float32)):
        pass  # engine h4 family not orthogonal; still test
    rot=(wflat.reshape(N,K//gs,gs)@R).reshape(N,K)
    am=np.abs(rot).max(axis=1,keepdims=True).clip(1e-30); sc=am/G**0*am/ (am.max()*0+am)*0+am/127.0
    sc=am/127.0
    q=T.from_numpy(rot/sc).round_().clamp_(-127,127).numpy().astype(np.int8)
    official_codes_i8=c.codes_i8(subj).reshape(N,K)
    eq=(q==official_codes_i8).mean()
    # also inverse rotation variant
    rot2=(wflat.reshape(N,K//gs,gs)@R.T).reshape(N,K)
    am2=np.abs(rot2).max(axis=1,keepdims=True).clip(1e-30); sc2=am2/127.0
    q2=T.from_numpy(rot2/sc2).round_().clamp_(-127,127).numpy().astype(np.int8)
    eq2=(q2==official_codes_i8).mean()
    print(f"{name}: codes match frac @R={eq:.4f}  @Rᵀ={eq2:.4f}")
