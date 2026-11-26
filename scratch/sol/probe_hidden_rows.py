import os, numpy as np
DEQ = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/dequant_dump_20261126"
DT = {"bf16": np.uint16, "f32": np.float32, "f64": np.float64}
def load(d, tag):
    with open(os.path.join(d, "manifest.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] != tag: continue
            rows, cols, dt, by = int(p[1]), int(p[2]), p[3], int(p[4])
            path = os.path.join(d, f"{tag}_{rows}x{cols}_{dt}.raw")
            if not os.path.exists(path):
                cands = [fn for fn in os.listdir(d) if fn.startswith(tag + "_") and fn.endswith(".raw")]
                path = os.path.join(d, cands[0])
            raw = np.fromfile(path, dtype=DT[dt], count=by // DT[dt]().itemsize)
            if dt == "bf16":
                return (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
            return raw.reshape(rows, cols)
pos = load(DEQ, "b0_layout_positions")
hid = load(DEQ, "b0_s0_hidden_in")
row_bad = (~np.isfinite(hid)).sum(axis=1)
row_inf = np.isinf(hid).sum(axis=1)
rows = hid.shape[0]
print("total rows:", rows, "cols:", hid.shape[1])
# classify by coords: video rows have 3 distinct positive coords; audio/cond special
uniq = {}
for i in range(rows):
    key = tuple(int(x) for x in pos[i])
    cat = "vid" if key[0] >= 0 and key[1] >= 0 and key[2] >= 0 and not (key[0] == -1 or key[1] == -1 or key[2] == -1) else "aux"
# simpler: group by (t) value ranges
tvals = pos[:, 0]
import collections
pat = collections.Counter()
for i in range(rows):
    bad = row_bad[i] > 0
    inf = row_inf[i] > 0
    t = int(pos[i][0]); h = int(pos[i][1]); w = int(pos[i][2])
    sig = (t < 0 or h < 0 or w < 0, bad, inf)
    pat[sig] += 1
print("(aux_row, has_nan, has_inf) -> count")
for k, v in sorted(pat.items()): print("  ", k, v)
# print boundaries: which contiguous row ranges are bad
bad_rows = np.nonzero(row_bad > 0)[0]
if len(bad_rows):
    # ranges
    splits = np.split(bad_rows, np.nonzero(np.diff(bad_rows) > 1)[0] + 1)
    print("bad row ranges (first 15):", [(int(r[0]), int(r[-1])) for r in splits[:15]])
    inf_rows = np.nonzero(row_inf > 0)[0]
    if len(inf_rows):
        sp2 = np.split(inf_rows, np.nonzero(np.diff(inf_rows) > 1)[0] + 1)
        print("inf row ranges (first 15):", [(int(r[0]), int(r[-1])) for r in sp2[:15]])
print("sample coords bad:", pos[bad_rows[:5]].tolist() if len(bad_rows) else "none")
print("sample coords good:", pos[[i for i in range(rows) if row_bad[i]==0][:5]].tolist())
