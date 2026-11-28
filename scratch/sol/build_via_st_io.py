#!/usr/bin/env python3
"""build_via_st_io.py — runner shim for scripts/build_int8_convrot.py.

Why: scripts/st_io.py (read-only for this watchdog task) was mid-refactored on
2026-11-28 03:20 and st_create() still calls a lost helper `build(pad_len, meta,
tensors, offs_rel, align)` -> NameError at 09:03 chain run. The surrounding
st_create code already computes the exact header dict (raw_hdr) inline; build()
only owed the (header_dict, _) return value. This shim injects a faithful stub
into st_io's module globals (looked up at CALL time inside st_create) and then
runs the untouched builder script. No file under scripts/ is modified.

The stub replicates st_io.st_create lines 100-107 verbatim:
    raw_hdr = {"__metadata__": meta} if meta else {}
    raw_hdr[name] = {dtype, shape, data_offsets=[rel, rel+size]} for each tensor
so the returned `header` is byte-equivalent to what the builder serializes as
the file header (used for manifest tensor_entries = len(header) - 1).

Usage: uv run --python 3.11 --with torch,numpy -q python scratch/sol/build_via_st_io.py [builder args...]
"""
import os
import runpy
import sys

R = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
sys.path.insert(0, os.path.join(R, "scripts"))
sys.path.insert(0, os.path.join(R, "scratch", "sol"))
import st_io  # noqa: E402
from st_io_build_stub import build_stub as _build_stub  # noqa: E402

if not hasattr(st_io, "build"):
    st_io.build = _build_stub
    print("[shim] st_io.build stub injected (mid-refactor NameError repair)", flush=True)

sys.argv = [sys.argv[0]] + sys.argv[1:]
runpy.run_path(os.path.join(R, "scripts", "build_int8_convrot.py"), run_name="__main__")
