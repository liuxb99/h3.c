import numpy, glob, os, sys
# Ground truth: recompute cond GEMM on CPU from dumped e3 taps, compare to GPU out.
d = sys.argv[1]
def find(pat):
    hits = sorted(glob.glob(os.path.join(d, f"*{pat}*.raw")))
    return hits[0] if hits else None
def load(p, shape=None):
    a = numpy.fromfile(p, dtype=numpy.uint16)
    f = (a.astype(numpy.uint32) << 16).view(numpy.float32)
    return f if p else None
sw, sb, ss, so = (find("e3_condition_w"), find("e3_condition_b"),
                  find("e3_source"), find("e1_after_cond"))
if not (sw and sb and ss and so):
    print("missing taps", {k: bool(v) for k,v in zip(("w","b","s","o"), (sw,sb,ss,so))}); sys.exit(1)
w, b, s, o = load(sw), load(sb), load(ss), load(so)
print("shapes: w", w.size, "b", b.size, "s", s.size, "o", o.size)
rows = s.size // 5120
w_m = w.reshape(5376, 5120)
print("w finite:", numpy.isfinite(w).all(), " b finite:", numpy.isfinite(b).all(),
      " s finite:", numpy.isfinite(s).all(), " out finite:", numpy.isfinite(o).all())
print("out nan frac:", float(numpy.isnan(o).mean()))
s_m = s.reshape(rows, 5120)
ref = s_m @ w_m.T + b[:5376]
ref16 = ((ref.view(numpy.uint32) >> 16) << 16).view(numpy.float32)  # truncate to bf16 grid
o_m = o.reshape(-1, 5376)[:rows]
rel = numpy.abs(o_m - ref16) / (numpy.abs(ref16) + 1e-8)
print("CPU ref finite:", numpy.isfinite(ref16).all())
print("max |GPUout - CPUref|:", float(numpy.abs(o_m - ref16).max()))
print("mean rel err:", float(rel[numpy.isfinite(rel)].mean()) if numpy.isfinite(rel).any() else "n/a")
# also: does out look like gemm WITHOUT dequant-requant w, i.e. using original golden file bytes?
