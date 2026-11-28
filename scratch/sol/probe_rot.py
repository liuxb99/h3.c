import sys, os, numpy as np, torch
sys.path.insert(0,'scripts')
torch.set_num_threads(14)
from qkv_layout_gate import WeightSource
c=WeightSource("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors")
r=WeightSource("/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer")

def h_kron4(G):
    """Engine hadamard_transposed H (before its final transpose): kron^log4(h4)."""
    h4 = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
    H = np.array([[1]],dtype=np.float32)
    while H.shape[0] < G:
        H = (H[:,:,None,None]*h4[None,None,:,:]).reshape(H.shape[0]*4,H.shape[0]*4)
    return H

def rot_kron4(x, G):
    H = h_kron4(G)
    N = x.shape[0]
    return (x.reshape(N,-1,G) @ H.T).reshape(N,-1)  # v_row @ H (H symmetric -> @H.T == @H)

def sylv(H):
    n=H.shape[0]
    return H

def cos(a,b): return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))

subj="blocks.10.mlp.fc2.weight"
gs,_=c.groupsize(subj)
shape=c.ent(subj)[1]["shape"]; print("official fc2 weight shape:",shape,"scale shape:",c.ent(subj[:-7]+".weight_scale")[1]["shape"])
N,K=shape
codes=c.codes_i8(subj).reshape(N,K).astype(np.float32); scale=c.scale_f32(subj)
x=codes*scale
H=h_kron4(gs); print("H symmetric:", np.array_equal(H,H.T), "H@H==I*G:", np.allclose(H@H, gs*np.eye(gs,dtype=np.float32)))
w=r.rows_f32(subj,0,N)
d1=(x.reshape(N,-1,gs)@H).reshape(N,K)   # v@H
print("cos(v@H dequant, HF)      =",cos(d1,w))
print("cos(raw, HF)              =",cos(x,w))
from convrot_quant import rotate_stages_f32
d2=rotate_stages_f32(torch.from_numpy(x.copy())).numpy().reshape(N,K)
print("cos(sylverson_dequant,HF) =",cos(d2,w))
# bit-reversal permuted sylvester variants
print("done")
