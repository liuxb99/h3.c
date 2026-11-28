import json, struct, os, sys, numpy as np, collections
root = sys.argv[1] if len(sys.argv) > 1 else "/Users/cyliu99/AI/h3c_mac_20261125/model_native"
out = []
for dp, dns, fns in os.walk(root):
    for fn in sorted(fns):
        if not fn.endswith(".safetensors"): continue
        p = os.path.join(dp, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n)); base = 8 + n
            rel = os.path.relpath(p, root)
            bad = []
            for k, v in hdr.items():
                if k == "__metadata__": continue
                dt = v["dtype"]
                if dt not in ("F32", "BF16", "F16"): continue
                o = v["data_offsets"]
                f.seek(base + o[0])
                nbytes = o[1] - o[0]
                npdt = np.float32 if dt == "F32" else np.uint16
                arr = np.frombuffer(f.read(nbytes), dtype=npdt)
                if dt != "F32":
                    arr = (arr.astype(np.uint32) << 16).view(np.float32)
                nan = int(np.isnan(arr).sum()); inf = int(np.isinf(arr).sum())
                if nan or inf:
                    bad.append((k, dt, tuple(v["shape"]), nan, inf))
            print(f"== {rel}: tensors scanned, bad={len(bad)}", flush=True)
            for b in bad[:20]:
                print("   BAD", b, flush=True)
print("SCAN DONE")
