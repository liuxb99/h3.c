import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# force fresh import of the core under test without its selfcheck aborting
import importlib.util
spec = importlib.util.spec_from_file_location("cq", "scripts/convrot_quant.py")
cq = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(cq)
    print("module import OK (selfcheck passed)")
except AssertionError as e:
    print("SELFCHECK FAILED:", e)

# brute force: which of {h4 kron build, sylvester} is symmetric/orthogonal?
h4 = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=np.float32)
def kron_build(G):
    H=np.array([[1.0]],dtype=np.float32)
    while H.shape[0]<G:
        W=H.shape[0]; grown=np.empty((W*4,W*4),dtype=np.float32)
        for i in range(W):
            for j in range(W): grown[i*4:(i+1)*4,j*4:(j+1)*4]=H[i,j]*h4
        H=grown
    return H
def sylv(G):
    H=np.ones((1,1),dtype=np.float32)
    while H.shape[0]<G: H=np.block([[H,H],[H,-H]])
    return H
for name,H in (("kron4",kron_build(16)),("sylv",sylv(16))):
    print(name,"symmetric:",np.array_equal(H,H.T),"ortho:",np.array_equal(H@H.T,16*np.eye(16,dtype=np.float32)))
# digit-stage transpose selfcheck at G=16 directly vs kron
from importlib import import_module
