# CPU oracle: dequantize every quantized tensor in the convrot int8 file
# (int8 * row_scale, inverse Hadamard for convrot tensors) and compare against
# the original bf16 FL2VA pack tensors. Labels the tensor that breaks.
import json, struct, sys
import numpy as np

INT8_FILE = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
ORIG_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/transformer"

def load_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    return f, hdr, 8 + n

def tensor(f, hdr, base, name):
    v = hdr[name]
    o = v["data_offsets"]
    f.seek(base + o[0])
    dt = v["dtype"]
    npdt = {"F32": np.float32, "BF16": np.uint16, "I8": np.int8, "U8": np.uint8, "F8_E4M3": np.uint8}[dt]
    item = {"F32": 4, "BF16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1}[dt]
    n_el = int(np.prod(v["shape"]))
    arr = np.frombuffer(f.read(n_el * item), dtype=npdt)
    if dt == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.reshape(v["shape"])

f, hdr, base = load_header(INT8_FILE)
meta = hdr.get("__metadata__", {})
print("file metadata keys:", list(meta.keys())[:8])

# original pack: merge all shards' headers
import glob, os
orig = {}
orig_f = {}
for p in sorted(glob.glob(os.path.join(ORIG_DIR, "*.safetensors"))):
    fo, ho, bo = load_header(p)
    for k in ho:
        if k != "__metadata__":
            orig[k] = (p, ho[k])

def hadamard(n):
    # exact transcription of h3_weights.c hadamard_transposed():
    # iterative kron with h4 = XOR-convolution Walsh core (h4[a][b] = -1 iff a XOR b == 3),
    # then the code transposes; the kernel consumes out[lane] = sum_k val[k] * M[k][lane],
    # i.e. dequant_group = val @ M with M = kron^4(h4), scaled by 1/sqrt(n) in-kernel.
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    M = np.array([[1.0]], dtype=np.float32)
    while M.shape[0] < n:
        M = np.kron(M, h4)
    return M

def stem_of(name):
    return name[:-7] if name.endswith(".weight") else name

qkeys = []
for k in sorted(hdr):
    if not (isinstance(hdr[k], dict) and hdr[k].get("dtype") == "I8"):
        continue
    stem = stem_of(k)
    if stem + ".weight_scale" in hdr and stem + ".comfy_quant" in hdr:
        qkeys.append(k)
print("quantized tensors:", len(qkeys))
results = []
for k in qkeys:
    stem = stem_of(k)
    if True:
        w = tensor(f, hdr, base, k).astype(np.float32)
        s = tensor(f, hdr, base, stem + ".weight_scale").astype(np.float32)
        sidecar = json.loads(tensor(f, hdr, base, stem + ".comfy_quant").tobytes().decode())
    convrot = bool(sidecar.get("convrot", False)) or "convrot" in json.dumps(sidecar)
    deq = w * s.reshape(w.shape[0], 1) if s.ndim == 1 or s.shape[1] == 1 else w * s
    # rotation block size comes from metadata (convrot_groupsize: adaln=64,
    # main attn/mlp=256); the kernel scales by 1/sqrt(groupsize).
    g = int(sidecar.get("convrot_groupsize", 256))
    if convrot and g > 1 and w.shape[1] % g == 0:
        H = hadamard(g)
        rows, cols = w.shape
        deq = (deq.reshape(rows, cols // g, g) @ H).reshape(rows, cols) / np.sqrt(g)
    if "attn.qkv_proj" in stem:
        # comfy files store q/k/v row blocks; the original pack is per-head
        # interleaved (q|k|v consecutive per head). Unpermute before compare.
        rps = w.shape[0] // 3
        heads = rps // 128
        if heads * 128 == rps and "token_refiner" not in stem:
            deq = deq.reshape(3, heads, 128, w.shape[1]).transpose(1, 0, 2, 3).reshape(w.shape)
    nan = int(np.isnan(deq).sum())
    if nan:
        results.append((k, "NaN_in_dequant", nan, 0.0, 0.0))
        continue
    # find original
    ok = None
    if k in orig:
        ok = k
    else:
        alt = k.replace(".weight", ".weight")
        for cand in (k, alt, k.replace("diffusers_", ""), "diffusers_" + k):
            if cand in orig:
                ok = cand
                break
    if ok is None:
        results.append((k, "no_original", 0, 0.0, 0.0))
        continue
    p2, v2 = orig[ok]
    fo, ho, bo = load_header(p2)
    o = tensor(fo, ho, bo, ok)
    if o.shape != deq.shape:
        results.append((k, "shape_mismatch", 0, 0.0, 0.0))
        continue
    d = np.abs(o - deq)
    scale_o = max(np.abs(o).max(), 1e-9)
    results.append((k, "compared", 0, float(d.max()), float(d.max() / scale_o)))

results.sort(key=lambda r: (-r[2], -r[4]))
print("%-70s %-16s %8s %10s %10s" % ("tensor", "status", "nan", "maxabs", "maxrel"))
for r in results[:40]:
    print("%-70s %-16s %8d %10.3e %10.3e" % (r[0][:70], r[1], r[2], r[3], r[4]))
worse = [r for r in results if r[1] in ("NaN_in_dequant", "shape_mismatch") or r[4] > 0.05]
print("SUSPECT COUNT:", len(worse))
for r in worse[:20]:
    print("SUSPECT", r)
print("ORACLE_DONE")
