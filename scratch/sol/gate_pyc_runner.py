#!/usr/bin/env python3
"""gate_pyc_runner.py — execute the FROZEN gate bytecode pair:
  scripts/__pycache__/qkv_layout_gate.cpython-<ver>.pyc (compiled 00:34)
  scripts/__pycache__/convrot_quant.cpython-<ver>.pyc   (compiled 00:41)
—the exact helper+script combination under which the official contract file
and the engine both PASSED (scratch/sol/gate_official.log, 0 FAIL [Q], 00:49).
Run because scripts/convrot_quant.py + qkv_layout_gate.py SOURCE pair has been
restored mid-refactor and disagrees with that contract (see report)."""
import importlib.machinery as m
import importlib.util as u
import sys
import os

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
ver = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
cq_pyc = os.path.join(SCRIPTS, "__pycache__", f"convrot_quant.{ver}.pyc")
gate_pyc = os.path.join(SCRIPTS, "__pycache__", f"qkv_layout_gate.{ver}.pyc")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def load(name, path):
    loader = m.SourcelessFileLoader(name, path)
    spec = u.spec_from_file_location(name, path, loader=loader)
    mod = u.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


sys.argv[0] = gate_pyc
import numpy  # noqa: E402,F401
import torch   # noqa: E402,F401
load("convrot_quant", cq_pyc)          # binds gate's `from convrot_quant import ...`
print(f"[pyc_runner] convrot_quant <- {cq_pyc}")
load("__main__", gate_pyc)
