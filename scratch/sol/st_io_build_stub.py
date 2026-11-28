"""st_io_build_stub.py — faithful stub for the lost st_io.build() helper.

Replicates st_io.st_create lines 100-107 (raw_hdr assembly) verbatim; returns
(header_dict, pad_len) as st_create expects from `build(pad_len, meta, tensors,
offs_rel, align)`. Shared by build_via_st_io.py (runner shim) and
smoke_st_io.py (synthetic round-trip test).
"""


def build_stub(pad_len, meta, tensors, offs_rel, align):
    raw_hdr = {}
    if meta:
        raw_hdr["__metadata__"] = meta
    for name, dtype, shape, size, _ in tensors:
        raw_hdr[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [offs_rel[name], offs_rel[name] + size]}
    return raw_hdr, pad_len
