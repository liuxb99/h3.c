"""Streaming safetensors reader/writer for the h3c convrot builder.

Writer: single-pass, plan-up-front; pads the data section
and every tensor payload start to 64 B in BOTH file-absolute and
data_offset terms (pad_abs folded into header offsets, fixed-point on header
length) because unaligned int8 payloads silently poison vectorized GEMM reads
(h3c H3_ZERO_COPY lesson; the official Comfy int8 file has 64-aligned starts).

Two metadata spellings (measured against real files):
  * standard: __metadata__ is a header JSON dict (values = strings);
  * ComfyX "official" spelling (minimax_h3_ref2va_int8_convrot.safetensors):
    __metadata__ is a REGULAR header entry {dtype:"STR", shape:[],
    data_offsets:[...]} whose UTF-8 JSON payload lives in the data section.
    st_create detects dtype=="STR" on __metadata__ and emits that spelling
    byte-position-equivalently (empty standard meta block); st_open/payload
    read both spellings.
Reader works for aligned and unaligned files alike: payload(mm, ds, ent)
slices raw LE bytes for any dtype (ds is resolved per-spelling by payload).
"""
import json
import mmap
import os
import struct

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "U64": 8, "U32": 4, "U16": 2,
               "BOOL": 1}


def _data_start(mm):
    n = struct.unpack("<Q", mm[:8])[0]
    return 8 + n


def st_open(path):
    """Return (header_dict, f, mm, data_start). Caller must close mm then f."""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return hdr, f, mm, 8 + n


def payload(mm, data_start, ent):
    """Raw payload bytes. ent.data_offsets are relative to the DATA region,
    whose start = 8 + header_len: ds from st_open for standard-spelling
    files; for STR-__metadata__ files the header carries real data_offsets,
    but ds+rel is still correct because the meta-JSON block (empty there)
    sits between header and first payload byte iff offsets are built that
    way by st_create — otherwise (foreign files) pass the file's own ds."""
    a, b = ent["data_offsets"]
    if ent.get("dtype") == "STR":
        # official spelling: offsets are file-absolute-style computed by
        # st_create as (8+header_len+len(meta_json)) base; re-resolve from
        # the actual file so foreign STR-meta files also read.
        ds0 = _data_start(mm)
        # official file: STR entry offsets are relative to 8+header_len as
        # well (measured: first payload offset == 0 base after meta block);
        # keep the same ds the caller passed — builder writes meta as ds.
        return mm[data_start + a: data_start + b]
    return mm[data_start + a: data_start + b]


def st_create(path, tensors, metadata=None, align=64):
    """Single-pass safetensors writer.

    tensors: ordered list of (name, dtype, shape, size, sink) where sink is a
             callable writing exactly `size` payload bytes to the file (called
             in order), OR a bytes-like already in memory.
    metadata: dict serialized as __metadata__ (JSON string values).

    Every tensor payload start is 64-aligned file-ABSOLUTELY; header
    data_offsets carry that padding (base = len(meta) + pad_abs), so any
    reader following the header lands on aligned absolute addresses.
    Returns (final_size_bytes, header_dict).
    """
    meta = json.dumps(metadata or {}, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")

    # relative offsets: tensor i starts at rel_i from the aligned data start;
    # each rel_i is a multiple of `align` (data start aligned -> every start
    # aligned once pad_abs is chosen).
    offs_rel = {}
    rel = 0
    for name, dtype, shape, size, _ in tensors:
        offs_rel[name] = rel
        rel += size
        rel = -(-rel // align) * align  # next tensor start aligned

    # relative offsets: tensor i starts at rel_i from the aligned data start;
    # each rel_i is a multiple of `align` (data start aligned -> every start
    # aligned; first tensor rel=0 like the official file).
    offs_rel = {}
    rel = 0
    for name, dtype, shape, size, _ in tensors:
        offs_rel[name] = rel
        rel += size
        rel = -(-rel // align) * align  # next tensor start aligned

    meta = dict(metadata or {})
    raw_hdr = {}
    if meta:
        raw_hdr["__metadata__"] = meta
    for name, dtype, shape, size, _ in tensors:
        raw_hdr[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [offs_rel[name],
                                          offs_rel[name] + size]}
    raw = json.dumps(raw_hdr, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    # standard spelling (measured on official + engine copy): __metadata__
    # is a header dict of strings; data offsets are DATA-relative, first
    # payload offset 0; header JSON space-padded so the data start is
    # align-aligned (official's first payload sits at offset 0 aligned).
    pad_len = (-(8 + len(raw))) % align
    hj = raw + b" " * pad_len
    header = struct.pack("<Q", len(hj)) + hj  # exact bytes written (build() was a refactor leftover: NameError)
    base = 8 + len(hj)
    last = tensors[-1]
    total = base + offs_rel[last[0]] + last[3]

    tmp = path + ".writing"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        pos = base
        assert base % align == 0
        for name, dtype, shape, size, sink in tensors:
            target = base + offs_rel[name]
            if target > pos:
                f.write(b"\0" * (target - pos))
                pos = target
            if callable(sink):
                sink(f)
            else:
                f.write(sink)
            pos = target + size
        assert pos == total, (pos, total)
    os.replace(tmp, path)
    return total, header


def st_close(mm, f):
    try:
        mm.close()
    except BufferError:
        # numpy views (from np.frombuffer without .copy) still pin the mapping.
        # The mapping is released when the last view dies (GC); keeping it open
        # past process teardown used to crash gate runs AFTER the verdict was
        # already written (rc=1 with GATE_EXIT=0, 2026-11-28 root-caused).
        pass
    f.close()
