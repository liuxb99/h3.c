import os, numpy as np
D = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/e0_dump_20261126"
def load(tag):
    with open(os.path.join(D, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] != tag: continue
            rows, cols, dt, by = int(p[1]), int(p[2]), p[3], int(p[4])
            path = os.path.join(D, f"{tag}_{rows}x{cols}_{dt}.raw")
            if not os.path.exists(path):
                cands = [fn for fn in os.listdir(D) if fn.startswith(tag + "_") and fn.endswith(".raw")]
                if not cands: return None, None
                path = os.path.join(D, cands[0])
            item = {"bf16": 2, "f32": 4, "f64": 8, "u32": 4}.get(dt)
            n = by // item
            if rows: n = min(n, rows * cols)
            npdt = {"bf16": np.uint16, "f32": np.float32, "f64": np.float64, "u32": np.uint32}[dt]
            a = np.fromfile(path, dtype=npdt, count=n)
            if dt == "bf16": a = (a.astype(np.uint32) << 16).view(np.float32)
            if rows and cols: a = a.reshape(rows, cols)
            return a, dt
    return None, None
for tag in ["e0_video_input", "e0_audio_input", "e0_video_projected", "e0_audio_projected",
            "e0_refined_text", "e0_hidden_packed"]:
    a, dt = load(tag)
    if a is None:
        print(f"{tag:22s}: MISSING"); continue
    a2 = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
    badrow = (~np.isfinite(a2)).sum(axis=1)
    nan = int(np.isnan(a).sum()); inf = int(np.isinf(a).sum())
    fin = a[np.isfinite(a)]
    ranges = ""
    nz = np.nonzero(badrow)[0]
    if len(nz):
        sp = np.split(nz, np.nonzero(np.diff(nz) > 1)[0] + 1)
        ranges = " badrow:" + ",".join(f"{s[0]}-{s[-1]}" for s in sp[:8])
    mx = float(np.max(np.abs(fin))) if fin.size else float("nan")
    print(f"{tag:22s} shape={a.shape} {dt}: nan={nan:8d} inf={inf:7d} max={mx:.3e}{ranges}")
