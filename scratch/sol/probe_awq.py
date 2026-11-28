import sys
sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
from measure_lib import open_st
p = '/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
st = open_st(p)
ks = list(st['hdr'])
vis = [k for k in ks if 'visual' in k]
print('visual key count:', len(vis))
for k in vis:
    if 'blocks.0.attn' in k or 'merger' in k:
        print(k, st['hdr'][k]['dtype'], st['hdr'][k]['shape'], st['hdr'][k].get('offsets'))
lm = [k for k in ks if 'layers.0.self_attn' in k]
for k in lm:
    print(k, st['hdr'][k]['dtype'], st['hdr'][k]['shape'])
meta = st['hdr'].get('__metadata__')
if meta:
    import json as j
    md = j.loads(meta) if isinstance(meta, str) else meta
    for kk, vv in list(md.items())[:12]:
        print('META', kk, str(vv)[:120])
st['f'].close()
