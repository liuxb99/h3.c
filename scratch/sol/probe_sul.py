import sys
sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
from measure_lib import open_st
p = '/Volumes/SSD2T/models/SulphurAI-Sulphur-3/s3_w4a8.safetensors'
st = open_st(p)
ks = sorted(k for k in st['hdr'] if 'blocks.0.attn' in k or 'blocks.0.mlp' in k or 'codebook' in k.lower())
for k in ks[:16]:
    print(k, st['hdr'][k]['dtype'], st['hdr'][k]['shape'])
st['f'].close()
print('====te_orig/ref_name_of====')
src = open('/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/stages2.py').read()
i = src.find('def te_orig')
print(src[i:i+400])
i = src.find('def ref_name_of')
if i < 0:
    from measure_lib import ref_name_of
    import inspect
    print(inspect.getsource(ref_name_of))
else:
    print(src[i:i+400])
