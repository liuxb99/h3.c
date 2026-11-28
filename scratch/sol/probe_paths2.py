import sys, glob, os
sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
from measure_lib import open_st
print("== Comfy-Org dirs")
for p in glob.glob('/Volumes/SSD2T/models/*'):
    if 'comfy' in os.path.basename(p).lower() or 'org' in os.path.basename(p).lower():
        print(p)
        try:
            print("  ", sorted(os.listdir(p))[:8])
        except Exception as e:
            print("  ", e)
        for sub in glob.glob(p + '/**/*.safetensors', recursive=True)[:12]:
            print("   FILE:", sub)
print("== VPIPE header")
st = open_st('/Volumes/SSD2T/models/VPipe-comfy_h3/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors')
ks = list(st['hdr'])
for k in [k for k in ks if 'blocks.0.attn.qkv_proj' in k]:
    print(k, st['hdr'][k]['dtype'], st['hdr'][k]['shape'])
scales = [k for k in ks if 'scale' in k.lower()][:4]
print('sample scale keys:', scales, {k: (st['hdr'][k]['dtype'], st['hdr'][k]['shape']) for k in scales})
refs = [k for k in ks if 'token_refiner' in k and ('qkv' in k or 'fc1' in k)][:4]
print('refiner keys:', refs, {k: st['hdr'][k]['dtype'] for k in refs})
st['f'].close()
