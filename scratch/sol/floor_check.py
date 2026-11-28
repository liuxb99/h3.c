import numpy as np, sys
sys.path.insert(0, "scratch/sol")
import oracle_independent as O

cf, chdr, CDS = O.open_st(O.CAND)
subj = "blocks.49.adaln_proj.linear.weight"
se = chdr[subj + ".weight_scale"]
ce = chdr[subj]
N = ce["shape"][0]
sc = O.pread_np(cf, CDS + se["data_offsets"][0], se["data_offsets"][1] - se["data_offsets"][0], np.float32).reshape(-1)
codes = O.pread_np(cf, CDS + ce["data_offsets"][0], N * ce["shape"][1], np.int8).reshape(N, -1)
fl = np.where(sc == O.FLOOR)[0]
print("N", N, "floor_count", len(fl), "range", fl.min(), fl.max(),
      "contig_bottom", bool((fl == np.arange(N - len(fl), N)).all()))
print("codes_zero_on_floor", bool((codes[fl] == 0).all()))
print("scale_floor_exact", int((sc == O.FLOOR).sum()), "min_nonfloor", float(np.min(sc[sc > O.FLOOR])))
print("frac", len(fl) / N)
near = np.nonzero(sc > O.FLOOR)[0].min()
print("first_nonfloor_row", near, "rows_above_bottom", N - near)
