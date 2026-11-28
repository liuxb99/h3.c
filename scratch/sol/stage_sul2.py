# standalone Sulphur-3 w4a8 dequant + row-order probes (CPU-only, read-only)
import sys, json
import numpy as np
sys.path.insert(0, '/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol')
from measure_lib import open_st, hf_tensor, cos_sim
from st_scan import grouped_from_interleaved, raw, fp8_e4m3_to_f32

SUL = '/Volumes/SSD2T/models/SulphurAI-Sulphur-3/s3_w4a8.safetensors'
OUT = open('/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/results2_SUL2.jsonl', 'w')

st = open_st(SUL)

def deq(name, rel_mode):
    packed = np.frombuffer(bytes(raw(st, name)), np.uint8).reshape(st['hdr'][name]['shape'])
    cb = np.frombuffer(bytes(raw(st, name + '_codebook')), np.float32)
    assert cb.shape == (16,), cb.shape
    sc = np.frombuffer(bytes(raw(st, name + '_s_channel')), np.float32)
    er = st['hdr'][name + '_s_rel']
    rb = bytes(raw(st, name + '_s_rel'))
    rel = (fp8_e4m3_to_f32(np.frombuffer(rb, np.uint8)) if rel_mode == 'e4m3'
           else np.frombuffer(rb, np.uint8).astype(np.float32)).reshape(er['shape'])
    lo = cb[packed & 15]; hi = cb[(packed >> 4) & 15]
    X = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
    X[:, 0::2] = lo; X[:, 1::2] = hi
    relb = np.repeat(rel, X.shape[1] // rel.shape[1], axis=1)
    assert relb.shape == X.shape, (relb.shape, X.shape)
    return X * relb * sc[:, None]

def slab_unit(M):
    A = M.reshape(M.shape[0] // 128, -1)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    return A

def slab_match(X, R):
    ux, ur = slab_unit(X), slab_unit(R)
    C = ux @ ur.T
    best = C.max(axis=1); who = C.argmax(axis=1)
    return float(best.mean()), float((who == np.arange(len(who))).mean())

def emit(key, X, refs):
    rec = {'stage': 'SUL2', 'key': key, 'shape': list(X.shape)}
    for label, ref in refs:
        m, idf = slab_match(X, ref)
        rec[label + '_slabmean'] = round(m, 4); rec[label + '_idfrac'] = round(idf, 4)
        rec[label + '_cos'] = round(cos_sim(X, ref), 5)
    OUT.write(json.dumps(rec) + '\n'); OUT.flush()

for i in (0, 25, 49):
    for rel_mode in ('i8', 'e4m3'):
        n = f'blocks.{i}.attn.qkv_proj.weight'
        R = hf_tensor(n)
        X = deq(n, rel_mode)
        emit(n + '|rel=' + rel_mode, X, (('vs_I', R), ('vs_G', grouped_from_interleaved(R))))
        del X
        n = f'blocks.{i}.mlp.fc1.weight'
        R = hf_tensor(n)
        X = deq(n, rel_mode)
        hs = np.concatenate([R[R.shape[0] // 2:], R[:R.shape[0] // 2]])
        emit(n + '|rel=' + rel_mode, X, (('vs_identity', R), ('vs_halfswap', hs)))
        del X, hs
    gc_needed = True
OUT.close(); st['f'].close()
print('SUL2_DONE')
