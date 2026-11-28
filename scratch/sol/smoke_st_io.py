#!/usr/bin/env python3
"""smoke_st_io.py — synthetic round-trip test of st_create with the injected
build() stub, before committing to a 34 GB double build. Writes only into
scratch/sol/. Checks: header correctness, payload bytes, 64 B absolute+relative
alignment, determinism (two runs byte-identical), st_open re-read consistency."""
import hashlib
import os
import sys

R = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
S = os.path.join(R, "scratch", "sol")
sys.path.insert(0, os.path.join(R, "scripts"))
sys.path.insert(0, S)
import st_io  # noqa: E402
from st_io_build_stub import build_stub  # noqa: E402

assert not hasattr(st_io, "build"), "st_io.build already present — shim not needed"
st_io.build = build_stub

META = {"config": '{"transformer": {"hidden_size": 8}}', "format": "pt"}
DATA = {
    "a.weight": ("I8", [3, 7], bytes(range(21))),            # 21 B, odd -> forces pad
    "a.weight_scale": ("F32", [3, 1], b"\x00" * 12),
    "a.comfy_quant": ("U8", [9], b'{"x": 123}'),
    "b.bias": ("BF16", [5], b"\x3f\x80" * 5),                 # 10 B
    "c.tab": ("F32", [16], b"\x01" * 64),                     # exact multiple
}


def plan_for(path_suffix):
    plan = []
    for name, (dt, shape, raw) in DATA.items():
        def sink_factory(raw=raw):
            def s(f):
                f.write(raw)
            return s
        plan.append((name, dt, shape, len(raw), sink_factory()))
    return plan


def one(path):
    total, header = st_io.st_create(path, plan_for(path), metadata=META, align=64)
    assert total == os.path.getsize(path), (total, os.path.getsize(path))
    return total, header


def check(path, header):
    h, f, mm, ds = st_io.st_open(path)
    assert ds % 64 == 0, ("data start not aligned", ds)
    assert h.get("__metadata__") == META
    for name, (dt, shape, raw) in DATA.items():
        ent = h[name]
        assert ent["dtype"] == dt and ent["shape"] == shape
        got = bytes(st_io.payload(mm, ds, ent))
        assert got == raw, (name, got, raw)
        a, b = ent["data_offsets"]
        assert a == 0 or a % 64 == 0, ("rel offset not aligned", name, a)
        assert (ds + a) % 64 == 0, ("abs start not aligned", name, ds + a)
        assert b - a == len(raw)
    # header entries returned == header entries serialized (shim fidelity)
    ser = {k: v for k, v in h.items()}
    assert header == ser, ("returned header != serialized header",
                           set(header) ^ set(ser))
    st_io.st_close(mm, f)
    # first tensor offset 0 (canonical layout claim)
    first = min(e["data_offsets"][0] for k, e in h.items() if k != "__metadata__")
    assert first == 0, first


p1, p2 = os.path.join(S, "smoke_st_a.safetensors"), os.path.join(S, "smoke_st_b.safetensors")
t1, h1 = one(p1)
t2, h2 = one(p2)
check(p1, h1)
check(p2, h2)
s1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
s2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
assert s1 == s2, "determinism failure"
# no stray .writing leftovers
assert not any(x.endswith(".writing") for x in os.listdir(S)), "stray .writing"
print("SMOKE PASS total=%d sha=%s (aligned abs+rel, header fidelity, determinism)" % (t1, s1[:16]))
for p in (p1, p2):
    os.remove(p)
