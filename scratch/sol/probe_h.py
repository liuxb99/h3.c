import numpy as np
def h_kron4(G):
    h4 = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
    H = np.array([[1.0]],dtype=np.float32)
    while H.shape[0] < G:
        H = (H[:,:,None,None]*h4[None,None,:,:]).reshape(H.shape[0]*4,H.shape[0]*4)
    return H
h4=np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
print("h4@h4 == 4I:", np.array_equal(h4@h4, 4*np.eye(4,dtype=np.float32)))
print("h4@h4.T:", )
A=h4@h4.T
print(A)
H=h_kron4(16); B=H@H.T
print("16 build offdiag max:", np.abs(B-np.diag(np.diag(B))).max(), "diag:", np.unique(np.diag(B)))
H4x=h_kron4(256); B4=H4x@H4x.T
print("256 offdiag max:", np.abs(B4-16*np.diag(np.diag(B4))//16).max() if False else np.abs(B4-np.diag(np.diag(B4))).max())
# try alternate: engine loop grows in-place using had (row-major) — replicate EXACTLY as in C
def h_engine(G):
    W=1; had=np.array([[1.0]],dtype=np.float32)
    while W<G:
        grown=np.empty((W*4,W*4),dtype=np.float32)
        for i in range(W):
            for j in range(W):
                for k in range(4):
                    for l in range(4):
                        grown[i*4+k][j*4+l]=had[i,j]*h4[k,l]
        had=grown; W*=4
    return had
HE=h_engine(16)
print("my build == engine build (16):", np.array_equal(h_kron4(16),HE))
print("HE16 orth:", np.array_equal(HE@HE.T, 16*np.eye(16,dtype=np.float32)))
