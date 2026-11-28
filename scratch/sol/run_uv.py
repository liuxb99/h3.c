#!/usr/bin/env python3
"""run_uv.py — uv-managed runner: ensure torch/numpy, exec target script with argv.

Parent uv venv is python-only (no pip, no torch); this runner creates its own
uv venv with --python from the parent venv (shared base interpreter, so the
`python` command resolves to the parent venv python inside it) and installs
torch/numpy via uv pip.
"""
import os
import subprocess
import sys

VENV = os.path.expanduser("~/.venvs/h3c-torch")
BASE_PY = sys.executable  # parent uv venv python


def ensure_venv():
    py = os.path.join(VENV, "bin", "python")
    if os.path.exists(py):
        return py
    subprocess.run(["uv", "venv", "--python", BASE_PY, VENV], check=True)
    subprocess.run(
        ["uv", "pip", "--python", py, "install", "-q", "torch", "numpy"],
        check=True)
    return py


def main():
    py = ensure_venv()
    os.execv(py, [py] + sys.argv[1:])


if __name__ == "__main__":
    main()
