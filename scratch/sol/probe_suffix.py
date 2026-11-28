#!/usr/bin/env python3
"""Probe sidecar key suffix convention (single vs double) in a safetensors header."""
import sys
import os

SCRIPTS = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scripts"
sys.path.insert(0, SCRIPTS)
import st_io as S

path = sys.argv[1]
hdr, f, mm, ds = S.st_open(path)
d = dict(hdr)
d.pop("__metadata__", None)
suf2 = sum(1 for k in d if k.endswith(".weight.comfy_quant"))
suf1 = len([k for k in d if k.endswith(".comfy_quant")]) - suf2
ws2 = sum(1 for k in d if k.endswith(".weight.weight_scale"))
ws1 = len([k for k in d if k.endswith(".weight_scale")]) - ws2
print(f"{os.path.basename(path)}: comfy_quant double={suf2} single={suf1} | weight_scale double={ws2} single={ws1}")
try:
    S.st_close(mm, f)
except BufferError:
    pass
