import json, numpy as np, struct, os, sys
p = os.path.realpath(sys.argv[1])
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); data_off = 8 + n
    cands = [k for k in hdr if any(t in k.lower() for t in ("time", "step", "freq", "mod"))]
    for name in sorted(cands):
        if name == "__metadata__":
            continue
        dtype, shape, off = hdr[name]["dtype"], hdr[name]["shape"], hdr[name]["data_offsets"]
        f.seek(data_off + off[0]); raw = f.read(off[1] - off[0])
        dt = {"BF16": np.uint16, "F32": np.float32, "I8": np.int8, "I32": np.int32,
              "F64": np.float64, "U8": np.uint8, "I64": np.int64}[dtype]
        a = np.frombuffer(raw, dtype=dt); info = f"{name:44s} {dtype:5s} {str(shape):16s}"
        if dtype == "BF16":
            v = (a.astype(np.uint32) << 16).view(np.float32)
            info += f" NaN={np.isnan(v).sum()}/{v.size} absmax={np.abs(v[~np.isnan(v)]).max():.3e}"
        elif dtype in ("F32", "F64"):
            a = a.astype(np.float64)
            info += f" NaN={np.isnan(a).sum()}/{a.size} absmax={np.abs(a[~np.isnan(a)]).max():.3e}"
        else:
            info += f" min={a.min()} max={a.max()}"
        print(info)
