#!/usr/bin/env python3
"""run_chain.py — pipeline orchestration with status sidecar (chain.sh runs it)."""
import datetime
import json
import os
import subprocess
import sys
import time

R = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
S = os.path.join(R, "scratch/sol")
BUILT_DIR = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built"
F = os.path.join(BUILT_DIR, "minimax_h3_ref2va_int8_convrot_h3c.safetensors")
UV = ["uv", "run", "--python", "3.11", "--with", "torch,numpy", "-q", "python"]
STATUS = os.path.join(S, "pipeline_status.json")
OFFICIAL = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"
ENGINE = "/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors"


def free_gb(path="/Volumes/SSD2T"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_bsize / 1e9


def load():
    try:
        return json.load(open(STATUS))
    except Exception:
        return {}


def save(d):
    d["updated_utc"] = datetime.datetime.utcnow().isoformat()
    tmp = STATUS + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, STATUS)


def run(tag, cmd, cwd=R, env_extra=None):
    log = os.path.join(S, f"chain_{tag}.log")
    env = dict(os.environ, GZIP="9")
    if env_extra:
        env.update(env_extra)
    with open(log, "ab") as fh:
        fh.write(f"\n===== {tag} start {datetime.datetime.utcnow().isoformat()}\n".encode())
        fh.flush()
        rc = subprocess.call(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        fh.write(f"===== {tag} rc={rc}\n".encode())
    st = load()
    st[tag] = {"rc": rc, "t": time.time()}
    save(st)
    print(f"[chain] {tag} rc={rc}", flush=True)
    return rc == 0


def main():
    st = load()
    steps = []

    def gate_built_vs_hf():
        return run("gate_built_vs_hf", UV + ["scripts/qkv_layout_gate.py",
                "--candidate", F, "--reference", HF_DIR,
                "--detail-out", os.path.join(S, "gate_built_vs_hf.json"),
                "--allow-missing-reference-keys"])

    def gate_engine_vs_hf():
        return run("gate_engine_vs_hf", UV + ["scripts/qkv_layout_gate.py",
                "--candidate", ENGINE, "--reference", HF_DIR,
                "--detail-out", os.path.join(S, "gate_engine_vs_hf.json"),
                "--allow-missing-reference-keys"])

    def gate_official_vs_hf():
        return run("gate_official_vs_hf", UV + ["scripts/qkv_layout_gate.py",
                "--candidate", OFFICIAL, "--reference", HF_DIR,
                "--detail-out", os.path.join(S, "gate_official_vs_hf.json"),
                "--allow-missing-reference-keys"])

    def repro_compare():
        return run("repro_compare", ["python3", "scratch/sol/repro_compare.py"])

    def compare_bytes():
        return run("compare_bytes", UV + ["scratch/sol/compare_bytes.py"])

    def record1():
        if not run("tensor_sha_build1", UV + ["scratch/sol/tensor_sha.py", F,
                     os.path.join(S, "built1_sha.json")]):
            return False
        return run("official_sha_new", ["shasum", "-a", "256", OFFICIAL],
                   env_extra=None)

    def build1():
        if free_gb() < 60:
            print(f"[chain] ABORT build1: free {free_gb():.0f}G < 60G", flush=True)
            return False
        return run("build1", UV + ["scratch/sol/build_via_st_io.py",
                                   "--out-dir", BUILT_DIR])

    def delete1():
        ok = True
        # /Volumes/SSD2T raises transient InterruptedError (see build_int8_convrot.ei_retry)
        for i in range(10):
            try:
                if os.path.exists(F):
                    os.remove(F)
                break
            except InterruptedError:
                if i == 9:
                    raise
                print(f"[chain] EINTR retry {i + 1}/10 in delete1", flush=True)
                time.sleep(0.25 * (i + 1))
        prev = load()
        prev["delete1"] = {"rc": 0, "t": time.time(), "free_gb": free_gb()}
        save(prev)
        return ok

    def build2():
        if free_gb() < 60:
            print(f"[chain] ABORT build2: free {free_gb():.0f}G", flush=True)
            return False
        return run("build2", UV + ["scratch/sol/build_via_st_io.py",
                                   "--out-dir", BUILT_DIR])

    def record2():
        return run("tensor_sha_build2", UV + ["scratch/sol/tensor_sha.py", F,
                     os.path.join(S, "built2_sha.json")])

    def parity():
        return run("parity", UV + ["scripts/build_int8_convrot.py", "--parity"])

    steps = [
        ("parity", parity),
        ("build1", build1),
        ("record1", record1),
        ("delete1", delete1),
        ("build2", build2),
        ("record2", record2),
        ("repro_compare", repro_compare),
        ("gate_built_vs_hf", gate_built_vs_hf),
        ("gate_engine_vs_hf", gate_engine_vs_hf),
        ("gate_official_vs_hf", gate_official_vs_hf),
        ("compare_bytes", compare_bytes),
    ]
    save({"chain_started": time.time(), "free_gb_at_start": free_gb()})
    failed = None
    for tag, fn in steps:
        ok = fn()
        if not ok:
            failed = tag
            break
    st = load()
    st["chain_done"] = True
    st["failed_step"] = failed
    st["free_gb_at_end"] = free_gb()
    save(st)
    print(f"[chain] DONE failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
