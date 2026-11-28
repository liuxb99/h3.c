#!/usr/bin/env python3
"""tensor_sha.py — whole-file + per-tensor sha256 manifest for a safetensors file.

Usage: tensor_sha.py <file.safetensors> <out_manifest.json>
Streams per-tensor payloads (mmap, no full-file load); prints whole-file sha
as well. Deterministic: manifest is {name: {dtype, shape, size, sha256}} plus
file-level fields; JSON dumped with sort_keys for reproducible diffs.
"""
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scripts")
from st_io import st_close, st_open  # noqa: E402


def main():
    path, out = sys.argv[1], sys.argv[2]
    h, f, mm, ds = st_open(path)
    per = {}
    total_bytes = 0
    names = sorted(k for k in h if k != "__metadata__")
    for i, name in enumerate(names):
        ent = h[name]
        a, b = ent["data_offsets"]
        sha = hashlib.sha256(mm[ds + a: ds + b])
        per[name] = {"dtype": ent["dtype"], "shape": ent["shape"],
                     "bytes": b - a, "sha256": sha.hexdigest()}
        total_bytes += b - a
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            print(f"{i + 1}/{len(names)} tensors hashed", flush=True)
    st_close(mm, f)
    whole = hashlib.sha256(open(path, "rb").read())
    doc = {"file": path, "size_bytes": os.path.getsize(path),
           "sha256": whole.hexdigest(), "n_tensors": len(per),
           "payload_bytes": total_bytes,
           "metadata": h.get("__metadata__"), "tensors": per}
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print("WHOLE_SHA", whole.hexdigest())
    print("wrote", out)


if __name__ == "__main__":
    main()
