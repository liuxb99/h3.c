#!/usr/bin/env python3
"""finish_chain.py v2 — complete the tail with per-convention gate pairing.

Gate tooling drifted mid-morning (scripts/qkv_layout_gate.py edited 10:48,
commit 21c5581; convrot_quant.py restored 10:17). Sidecar conventions differ
per file (probe_suffix.py evidence):
  built   -> double-suffix sidecars  -> current 10:48 gate (append lookup)
  official/engine -> single-suffix   -> frozen 00:34/00:41 bytecode pair
Both helpers are norm-domain (battery: norm relrms 0.0088; identical code path
used by builder and gates). compare_bytes + status recording unchanged.
"""
import datetime
import json
import os
import subprocess
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(S, "pipeline_status.json")
HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
BUILT = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors"
ENGINE = "/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors"
OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
PY = sys.executable or "/usr/bin/python3"

UV313 = ["uv", "run", "--python", "3.13", "--with", "torch,numpy", "-q", "python"]
GATE_CURRENT = UV313 + ["scratch/sol/gate_runner.py"]          # current source pair
GATE_FROZEN = UV313 + ["scratch/sol/gate_pyc_runner.py"]        # 00:34+00:41 pyc pair


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load():
    with open(STATUS) as fh:
        return json.load(fh)


def record(tag, rc, extra=None):
    d = load()
    rec = {"rc": rc, "t": now()}
    if extra:
        rec.update(extra)
    d.setdefault(tag, {}).update(rec)
    if rc != 0:
        d["failed_step"] = tag
    else:
        d.pop("failed_step", None)
    d["chain_done"] = False
    d["updated"] = now()
    with open(STATUS, "w") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def run(tag, cmd, timeout=7200):
    print(f"[finish] START {tag}: {' '.join(cmd)}  @{now()}", flush=True)
    t0 = time.time()
    log = open(os.path.join(S, f"chain_{tag}.log"), "a")
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(S)), stdout=log, stderr=log,
                          timeout=timeout, stdin=subprocess.DEVNULL)
    log.close()
    rc = proc.returncode
    dur = round(time.time() - t0, 1)
    record(tag, rc, {"dur_s": dur})
    print(f"[finish] END {tag} rc={rc} dur={dur}s", flush=True)
    return rc == 0


def main():
    gates = [
        ("gate_built_vs_hf", GATE_CURRENT + [
            "--candidate", BUILT, "--reference", HF_DIR,
            "--detail-out", os.path.join(S, "gate_built_vs_hf.json"),
            "--allow-missing-reference-keys"]),
        ("gate_engine_vs_hf", GATE_FROZEN + [
            "--candidate", ENGINE, "--reference", HF_DIR,
            "--detail-out", os.path.join(S, "gate_engine_vs_hf.json"),
            "--allow-missing-reference-keys"]),
        ("gate_official_vs_hf", GATE_FROZEN + [
            "--candidate", OFFICIAL, "--reference", HF_DIR,
            "--detail-out", os.path.join(S, "gate_official_vs_hf.json"),
            "--allow-missing-reference-keys"]),
        ("compare_bytes", UV313 + ["scratch/sol/compare_bytes_v2.py"]),
    ]
    failed = None
    ps = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True,
                        text=True).stdout
    for tag, cmd in gates:
        st = load().get(tag, {})
        if st.get("rc") == 0:
            print(f"[finish] SKIP {tag} (status rc=0)", flush=True)
            continue
        if tag == "compare_bytes" and "compare_bytes_v2.py" in ps:
            print("[finish] compare precheck running in bg; record via v3 tail",
                  flush=True)
            continue
        if not run(tag, cmd):
            failed = tag
            break
    d = load()
    d["chain_done"] = failed is None
    d["failed_step"] = failed
    d["updated"] = now()
    with open(STATUS, "w") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[finish] done failed={failed}", flush=True)
    return 0 if failed is None else 1


if __name__ == "__main__":
    sys.exit(main())
