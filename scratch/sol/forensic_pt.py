import json, struct, os, numpy as np
CAND="/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
GDIR="/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
K="blocks.0.norm1.weight"
with open(CAND,"rb") as f:
    ln=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(ln)); DS=8+ln
    c=hdr[K]; f.seek(DS+c["data_offsets"][0]); cb=f.read(c["data_offsets"][1]-c["data_offsets"][0])
idx=json.load(open(os.path.join(GDIR,"model.safetensors.index.json")))["weight_map"]
shard=os.path.join(GDIR, idx[K])
with open(shard,"rb") as g:
    ln2=struct.unpack("<Q",g.read(8))[0]; gh=json.loads(g.read(ln2)); DS2=8+ln2
    ge=gh[K]; g.seek(DS2+ge["data_offsets"][0]); gb=g.read(ge["data_offsets"][1]-ge["data_offsets"][0])
print("cand", c["dtype"], c["shape"], len(cb))
print("gold", ge["dtype"], ge["shape"], len(gb), os.path.basename(shard))
print("equal:", cb==gb, "first diff:", next((i for i,(a,b) in enumerate(zip(cb,gb)) if a!=b), None))
if c["dtype"]=="F32" and ge["dtype"]=="BF16":
    a=np.frombuffer(cb,dtype=np.float32)
    b=np.frombuffer(gb,dtype=np.uint16).view(np.uint16)
    # bf16 -> f32
    b32=(b.astype(np.uint32)<<16).view(np.float32)
    print("bf16->f32 equal:", np.array_equal(a,b32), "maxdiff:", np.abs(a-b32).max())
    # check f32->bf16 truncation vs rounding
    print("sample cand:", a[:6], "golden:", b32[:6])
