import os, numpy as np
D = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/e0b_dump_20261126"
def load(tag):
    fn = [f for f in os.listdir(D) if f.startswith(tag + "_") and f.endswith(".raw")]
    if not fn: return None
    path = os.path.join(D, fn[0])
    base = fn[0][:-4]
    rc, dt = base.rsplit("_", 1)[0].rsplit("_", 1)[1], base.rsplit("_", 1)[1]
    rows, cols = (int(x) for x in rc.split("x"))
    n = os.path.getsize(path) // {"bf16": 2, "f32": 4, "u32": 4}[dt]
    a = np.fromfile(path, dtype={"bf16": np.uint16, "f32": np.float32, "u32": np.uint32}[dt], count=n)
    if dt == "bf16": a = (a.astype(np.uint32) << 16).view(np.float32)
    if rows and cols: a = a[:rows * cols].reshape(rows, cols)
    return a
for tag in ["e1_text_source", "e1_after_cond", "e1_after_refiner0",
            "e1_after_refiner1", "e1_after_finalnorm",
            "e2_video_patch_w", "e2_video_patch_b", "e2_audio_patch_w",
            "e0_video_input", "e0_hidden_packed"]:
    a = load(tag)
    if a is None:
        print(f"{tag:22s}: MISSING"); continue
    a2 = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
    nan = int(np.isnan(a).sum()); inf = int(np.isinf(a).sum())
    fin = a[np.isfinite(a)]
    mx = float(np.max(np.abs(fin))) if fin.size else float("nan")
    nz = np.nonzero((~np.isfinite(a2)).sum(axis=1))[0]
    ranges = ""
    if len(nz):
        sp = np.split(nz, np.nonzero(np.diff(nz) > 1)[0] + 1)
        ranges = " badrow:" + ",".join(f"{s[0]}-{s[-1]}" for s in sp[:6])
    print(f"{tag:22s} shape={a.shape}: nan={nan:8d} inf={inf:7d} max={mx:.3e}{ranges}")
