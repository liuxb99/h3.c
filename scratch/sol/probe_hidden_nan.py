import numpy as np, os, sys, glob
D = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/native_dump_20261126"
files = {}
for f in glob.glob(D + "/*.raw"):
    stem = os.path.basename(f)[:-4]          # drop ".raw"
    tag, ext = stem.rsplit("_", 1)           # "..._2790x5376_bf16"
    files[tag] = f
def load(tag):
    f = next((v for k, v in files.items() if k == tag or k.startswith(tag + "_")), None)
    if f is None:
        return None
    ext = os.path.basename(f)[:-4].rsplit("_", 1)[1]
    if ext == "bf16":
        a = np.fromfile(f, dtype=np.uint16)
        a = (a.astype(np.uint32) << 16).view(np.float32)
    elif ext == "f32":
        a = np.fromfile(f, dtype=np.float32)
    elif ext == "f64":
        a = np.fromfile(f, dtype=np.float64)
    else:
        a = np.fromfile(f, dtype=np.int64 if ext == "u32" else np.float32)
    return a.astype(np.float64)
print("available input-side taps:", sorted(k for k in files if not k.startswith(("b0_s", "b1_s")) )[:40])
hi = load("b0_s0_hidden_in")
if hi is not None:
    n = hi.size
    # try reshape 2790x5376
    if n % 5376 == 0:
        H = hi.reshape(n // 5376, 5376)
        nanmask = ~np.isfinite(H)
        rows_nan = nanmask.sum(axis=1); cols_nan = nanmask.sum(axis=0)
        print("rows:", H.shape[0], "rows fully-NaN:", int((rows_nan == H.shape[1]).sum()),
              "rows clean:", int((rows_nan == 0).sum()))
        print("per-row NaN count: min", rows_nan.min(), "max", rows_nan.max(),
              "mean", rows_nan.mean())
        # column pattern of first few NaN rows
        idx = np.where(rows_nan > 0)[0][:3]
        for r in idx:
            cols = np.where(nanmask[r])[0]
            runs = np.split(cols, np.where(np.diff(cols) != 1)[0] + 1)
            print(f"row {r}: {len(cols)} NaN cols in {len(runs)} runs; first runs:",
                  [(int(r_[0]), int(r_[-1])) for r_ in runs[:6]])
        cols_any = np.where(cols_nan > 0)[0]
        if len(cols_any):
            runs = np.split(cols_any, np.where(np.diff(cols_any) != 1)[0] + 1)
            print("cols with any NaN:", len(cols_any), "in", len(runs), "runs; first:",
                  [(int(r_[0]), int(r_[-1])) for r_ in runs[:8]])
for t in ["step0_video_input_f32", "step0_denoise_input_latents", "step0_denoise_input_audio",
          "s0_time_embed", "adaln_mod_table"]:
    x = load(t)
    if x is None:
        print(t, "MISSING"); continue
    fin = np.isfinite(x)
    print(f"{t:34s} n={x.size:11d} nonfinite={int((~fin).sum()):9d} "
          f"absmax={np.abs(x[fin]).max() if fin.any() else 0:.3e}")
