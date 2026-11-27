# Compare refine-condition tensors between convrot file (model_native target) and comfy file.
import json, struct, sys
import numpy as np

CONVROT = "/Volumes/SSD2T/models/AI-h3-metal-MiniMax-H3/minimax_h3_ref2va_int8_convrot.safetensors"
COMFY = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"

def load_header(p):
    f = open(p, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    return f, hdr, 8 + n

def tensor(f, hdr, base, name):
    v = hdr[name]
    o = v["data_offsets"]
    f.seek(base + o[0])
    dt = v["dtype"]
    npdt = {"F32": np.float32, "BF16": np.uint16, "I8": np.int8, "U8": np.uint8}[dt]
    item = {"F32": 4, "BF16": 2, "I8": 1, "U8": 1}[dt]
    n_el = int(np.prod(v["shape"]))
    arr = np.frombuffer(f.read(n_el * item), dtype=npdt)
    if dt == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.reshape(v["shape"])

def keys_like(hdr, pat):
    return sorted(k for k in hdr if pat in k and k != "__metadata__")

def report(path):
    print("=" * 20, path)
    f, hdr, base = load_header(path)
    keys = keys_like(hdr, "condition")
    if not keys:
        keys = keys_like(hdr, "text")
    for k in keys:
        print("  KEY", k, hdr[k]["dtype"], hdr[k]["shape"])
    # dequantize the main condition weight tensor in both files if present
    for name in keys:
        if hdr[name]["dtype"] != "I8":
            continue
        scale_key = name + ".weight_scale"
        if scale_key not in hdr:
            print("  !! no scale for", name)
            continue
        w = tensor(f, hdr, base, name).astype(np.float32)
        s = tensor(f, hdr, base, scale_key).astype(np.float32)
        print(f"  {name}: scale_shape={s.shape} scale_nan={int(np.isnan(s).sum())} scale_inf={int(np.isinf(s).sum())} scale_absmax={np.abs(s).max():.3e}")
        # broadcast dequant: scales are per (row, group256) typically
        rows, cols = w.shape
        sg = s.reshape(s.shape[0], -1) if s.ndim > 1 else s.reshape(1, -1)
        groups = cols // sg.shape[1] if sg.shape[1] <= cols else None
        if groups and cols % sg.shape[1] == 0:
            deq = (w * np.repeat(sg, groups, axis=1))
        else:
            deq = w * s.reshape(-1)[:1]
        print(f"  DEQ {name}: nan={int(np.isnan(deq).sum())} inf={int(np.isinf(deq).sum())} maxabs={np.abs(deq).max():.3e} meanabs={np.abs(deq).mean():.3e}")
        np.save("/tmp/deq_cond_%s.npy" % name.replace(".", "_") + ("_convrot" if "convrot" in path and "/MiniMax-H3-comfy" not in path else "_comfy"), deq)

report(CONVROT)
report(COMFY)
print("COND_DIFF_DONE")
