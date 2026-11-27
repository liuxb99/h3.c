import json,struct,numpy as np,os,sys
p=os.path.realpath(sys.argv[1] if len(sys.argv)>1 else "model_native/FL2VA/transformer/model-00001-of-00001.safetensors")
f=open(p,"rb"); n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n)); base=8+n
def tensor(name):
    v=hdr[name]; o=v["data_offsets"]; a=base+o[0]
    shape=tuple(v["shape"]); dt=v["dtype"]
    n_el=int(np.prod(shape)); itemsize={'F32':4,'BF16':2,'I8':1,'U8':1}[dt]
    npdt={'F32':np.float32,'BF16':np.uint16,'I8':np.int8,'U8':np.uint8}[dt]
    f.seek(a); return np.frombuffer(f.read(n_el*itemsize),dtype=npdt).reshape(shape)
for w in ["blocks.22.attn.out_proj","blocks.22.mlp.fc1","blocks.22.adaln_proj"]:
    sc=tensor(w+".weight_scale"); q=tensor(w+".comfy_quant")
    print(f"{w}: scale shape={sc.shape} NaN={int(np.isnan(sc).sum())} inf={int(np.isinf(sc).sum())} absmax={np.nanmax(np.abs(sc)):.3e} min_nz={np.nanmin(np.abs(sc[sc!=0])):.3e}")
    print(f"   comfy_quant len={q.size} bytes[:16]={q[:16].tolist()}")
    i8=tensor(w+".weight")
    flat=i8.reshape(i8.shape[0],-1)
    rowabsmax=np.abs(flat).max(axis=1)
    print(f"   I8 shape={i8.shape} min={i8.min()} max={i8.max()} zero%={100*(i8==0).mean():.1f} rows_at_127={int((rowabsmax>=127).sum())}/{flat.shape[0]}")
EOF_MARKER_NOT_USED = None
