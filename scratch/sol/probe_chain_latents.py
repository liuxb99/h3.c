import os, numpy as np
DT = {"bf16": np.uint16, "f32": np.float32, "f64": np.float64, "u32": np.uint32}
def load(d, tag):
    with open(os.path.join(d, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] != tag: continue
            rows, cols, dt, by = int(p[1]), int(p[2]), p[3], int(p[4])
            path = os.path.join(d, f"{tag}_{rows}x{cols}_{dt}.raw")
            if not os.path.exists(path):
                cands = [fn for fn in os.listdir(d) if fn.startswith(tag + "_") and fn.endswith(".raw")]
                if not cands: return None
                path = os.path.join(d, cands[0])
            raw = np.fromfile(path, dtype=DT[dt], count=by // DT[dt]().itemsize)
            if dt == "bf16":
                return (raw.astype(np.uint32) << 16).view(np.float32)
            return raw
    return None
for tag in ["step0_denoise_input_latents", "s0_velocity_bf16", "s0_video_latent_f32", "video_latent_final_f32"]:
    for label, d in (("deq", "/Users/cyliu99/AI/h3c_mac_20261125/evidence/dequant_dump_20261126"),
                     ("nat", "/Users/cyliu99/AI/h3c_mac_20261125/evidence/native_dump_20261126"),
                     ("base", "/Users/cyliu99/AI/h3c_mac_20261125/evidence/fulldit_dump_baseline")):
        x = load(d, tag)
        if x is None:
            print(f"{tag:32s} {label}: MISSING"); continue
        nan = int(np.isnan(x).sum()); inf = int(np.isinf(x).sum())
        fin = x[np.isfinite(x)]
        stats = f"nan={nan:7d} inf={inf:6d}"
        if fin.size:
            stats += f" mean={np.mean(fin):+.4f} std={np.std(fin):.4f} min={np.min(fin):+.3e} max={np.max(fin):+.3e}"
        print(f"{tag:32s} {label}: {stats}")
