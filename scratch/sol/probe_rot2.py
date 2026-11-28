import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
from convrot_quant import rotate_stages_f32
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")

def h_kron4(G):
    h4 = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
    H = np.array([[1.0]],dtype=np.float32)
    while H.shape[0] < G:
        H = (H[:,:,None,None]*h4[None,None,:,:]).reshape(H.shape[0]*4,H.shape[0]*4)
    return H

def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))

subj="blocks.10.mlp.fc2.weight"
gs,_=c.groupsize(subj)
N,K=c.ent(subj)[1]["shape"]
codes=c.codes_i8(subj).reshape(N,K).astype(np.float32); scale=c.scale_f32(subj)
x=codes*scale
H=h_kron4(gs)
print("H@Hᵀ==I·G:",np.allclose(H@H.T, gs*np.eye(gs,dtype=np.float32)))
w=r.rows_f32(subj,0,N)
d_kT=(x.reshape(N,-1,gs)@H.T).reshape(N,K)
print("cos(v@Hᵀ(engine-kernel) , HF) =",cos(d_kT,w))
d_syl=rotate_stages_f32(torch.from_numpy(x.copy())).numpy().reshape(N,K)
print("cos(sylvester-butterfly , HF) =",cos(d_syl,w))
