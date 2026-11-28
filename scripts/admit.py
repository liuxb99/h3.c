#!/usr/bin/env python3
"""admit.py — whole-machine resource admission gate (R7, post-panic 2026-11-28).

Purpose: ONE arbiter for heavy work (model builds, native-engine GPU runs,
34GB diffs, heavy IO) vs production services (Strata, ComfyX). Heavy classes
are mutually exclusive by default. A retried/aborted job may not be admitted
until the previous registration is released AND the snapshot shows the machine
has actually recovered (swap growth stopped + pressure back in band) — Metal
allocations are not visible in ps RSS, so recovery is judged by whole-machine
signals, per measured-host policy.

Every run registers a UNIQUE run_id + pid + start-time; chains must record the
per-run rc artifact path — never judge a new task from grepping old logs.

CLI (stdlib-only, no deps):
  admit.py status                      snapshot + live registrations
  admit.py request CLASS --run-id R [--reason S] [--force]
        exit 0 ADMIT (registration written) | 3 BUSY (other heavy active)
        | 4 PRESSURE (denied by memory rules) | 5 GPU-CAP (resident GPU budget)
  admit.py release --run-id R [--rc N] [--artifact PATH]
  admit.py watch [--interval S]        sampler loop -> monitor JSONL + ALERT lines
Classes: light_cpu | heavy_build | heavy_io | heavy_gpu | big_diff
Config: ../admission/policy.json (auto-created with measured defaults).
Registry: ../admission/registry.jsonl  Monitor: ../admission/monitor.jsonl
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # h3c_mac_20261125
ADM = os.path.join(ROOT, "admission")
REG = os.path.join(ADM, "registry.jsonl")
MON = os.path.join(ADM, "monitor.jsonl")
POL = os.path.join(ADM, "policy.json")

DEFAULT_POLICY = {
    "classes": {
        "light_cpu":    {"est_gb": 4,   "min_free_pct": 25, "mutex": False},
        "heavy_io":     {"est_gb": 8,   "min_free_pct": 35, "mutex": True},
        "big_diff":     {"est_gb": 16,  "min_free_pct": 40, "mutex": True,
                         "require_streaming": True},
        "heavy_build":  {"est_gb": 14,  "min_free_pct": 45, "mutex": True},
        "heavy_gpu":    {"est_gb": 20,  "min_free_pct": 55, "mutex": True,
                         "gpu_gb": 80},
    },
    "gpu_cap_gb": 100.0,
    "resident_gpu_gb": {"strata": 0.0, "comfyx_service": 24.0,
                        "h3_native_int8": 80.0, "comfyx_active_pipeline": 40.0},
    "swap_used_warn_gb": 12.0,
    "swap_used_block_gb": 30.0,
    "swap_rate_block_gb_per_min": 2.0,
    "compressor_block_mb": 8000,
    "recover_wait_s": 120,
}

HEAVY = ("heavy_io", "big_diff", "heavy_build", "heavy_gpu")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=15).stdout
    except Exception:
        return ""


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_policy():
    os.makedirs(ADM, exist_ok=True)
    if not os.path.exists(POL):
        with open(POL, "w") as f:
            json.dump(DEFAULT_POLICY, f, indent=2)
        return dict(DEFAULT_POLICY)
    with open(POL) as f:
        p = json.load(f)
    merged = dict(DEFAULT_POLICY)
    merged.update(p)
    return merged


def snapshot(policy):
    s: Dict[str, Any] = {"ts": now_iso()}
    out = sh(["sysctl", "-n", "vm.swapusage"])
    m = re.search(r"used = (\d+.\d+)M", out)
    s["swap_used_gb"] = round(float(m.group(1)) / 1024, 2) if m else None
    vm = sh(["vm_stat"])
    ps = 16384  # 16k pages on this host (from memory_pressure banner)
    mpp = re.search(r"page size of (\d+)", sh(["memory_pressure", "-Q"]))
    if mpp:
        ps = int(mpp.group(1))
    def pages(pat):
        mm = re.search(pat + r":\s+(\d+)", vm)
        return int(mm.group(1)) if mm else 0
    for k, pat in (("free_gb", "Pages free"), ("speculative_gb", "Pages speculative"),
                   ("inactive_gb", "Pages inactive"), ("purgeable_gb", "Pages purgeable"),
                   ("wired_gb", "Pages wired down"), ("compressed_gb", "Pages occupied by compressor")):
        s[k] = round(pages(pat) * ps / 2**30, 2)
    out = sh(["memory_pressure", "-Q"])
    m = re.search(r"free percentage:\s*(\d+)%", out)
    s["free_pct"] = int(m.group(1)) if m else None
    # live heavy signals
    s["procs"] = {}
    for name, pat in (("strata", "strata"), ("comfyx", "ComfyUI/main.py"),
                      ("h3_native", "./h3 "), ("chain", "run_chain.py"),
                      ("gate", "qkv_layout_gate.py"), ("oracle", "oracle_independent.py")):
        pid = sh(["pgrep", "-f", pat]).split("\n")[0].strip()
        if pid:
            rss = sh(["ps", "-o", "rss=", "-p", pid]).strip()
            s["procs"][name] = {"pid": int(pid), "rss_gb": round(int(rss or 0) / 2**20, 2)}
    df = sh(["df", "-g", "/Volumes/SSD2T"]).split("\n")[-1].split()
    s["ssd_free_gb"] = int(df[3]) if len(df) > 3 else None
    return s


def resident_gpu_estimate(s, policy):
    est = 0.0
    r = policy["resident_gpu_gb"]
    if "comfyx" in s["procs"]:
        est += r["comfyx_service"]
        # active pipeline = worker/render child processes or queue churn: crude proxy
        out = sh(["pgrep", "-f", "ComfyUI/.venv"])
        if len(out.split()) > 2:
            est += r["comfyx_active_pipeline"]
    if "h3_native" in s["procs"]:
        est += r["h3_native_int8"]
    return est


def read_registry():
    regs: List[Dict[str, Any]] = []
    if os.path.exists(REG):
        with open(REG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        regs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    live = {}
    for r in regs:
        if r["event"] == "admit":
            live[r["run_id"]] = r
        elif r["event"] == "release":
            live.pop(r["run_id"], None)
    # gc dead registrations
    for rid, r in list(live.items()):
        try:
            os.kill(int(r["pid"]), 0)
        except (ProcessLookupError, ValueError, PermissionError):
            if time.time() - r.get("_ts_epoch", time.time()) > 60:
                live.pop(rid)
    return live


def append(event):
    os.makedirs(ADM, exist_ok=True)
    with open(REG, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def swap_rate(policy):
    """swap GB/min over last ~5 monitor samples; None if insufficient."""
    if not os.path.exists(MON):
        return None
    pts = []
    with open(MON) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("swap_used_gb") is not None:
                    pts.append((d.get("_epoch") or 0, d["swap_used_gb"]))
            except Exception:
                pass
    pts = pts[-6:]
    if len(pts) < 3:
        return None
    dt = pts[-1][0] - pts[0][0]
    if dt <= 0:
        return None
    return round((pts[-1][1] - pts[0][1]) * 60.0 / dt, 2)


def request(args):
    policy = load_policy()
    cls = args.cls
    c = policy["classes"][cls]
    s = snapshot(policy)
    live = read_registry()
    reasons = []

    # 1) mutex over heavy classes (and self)
    active_heavy = [(rid, r["cls"]) for rid, r in live.items()
                    if r["cls"] in HEAVY]
    if active_heavy and (cls in HEAVY or not c.get("mutex", True) is False):
        if cls in HEAVY:
            others = [(rid, k) for rid, k in active_heavy]
            if others and not args.force:
                return deny("BUSY", f"heavy active: {others}", s, policy, 3)

    # 2) memory band
    fp = s.get("free_pct")
    rate = swap_rate(policy)
    sw = s.get("swap_used_gb") or 0
    if fp is not None and fp < c["min_free_pct"]:
        reasons.append(f"free_pct {fp} < {c['min_free_pct']}")
    if sw > policy["swap_used_block_gb"]:
        reasons.append(f"swap used {sw}G > block {policy['swap_used_block_gb']}G")
    elif sw > policy["swap_used_warn_gb"] and cls in ("heavy_gpu", "heavy_build", "big_diff"):
        reasons.append(f"swap used {sw}G above warn {policy['swap_used_warn_gb']}G for {cls}")
    if rate is not None and rate > policy["swap_rate_block_gb_per_min"]:
        reasons.append(f"swap growth {rate}G/min > {policy['swap_rate_block_gb_per_min']}")
    if (s.get("compressed_gb") or 0) * 1024 > policy["compressor_block_mb"]:
        reasons.append(f"compressor {s['compressed_gb']}G over cap")
    # projected: current free minus class estimate must still clear warn band
    est_free = (s.get("free_gb", 0) + s.get("purgeable_gb", 0)
                + s.get("speculative_gb", 0) + s.get("inactive_gb", 0) * 0.6)
    if est_free - c["est_gb"] < policy["swap_used_warn_gb"]:
        reasons.append(f"projected free after +{c['est_gb']}G = {round(est_free - c['est_gb'],1)}G < warn")
    if reasons and not args.force:
        return deny("PRESSURE", "; ".join(reasons), s, policy, 4)

    # 3) GPU budget
    if cls == "heavy_gpu":
        rg = resident_gpu_estimate(s, policy)
        if rg + c.get("gpu_gb", 0) > policy["gpu_cap_gb"] and not args.force:
            return deny("GPU-CAP",
                        f"resident GPU ~{rg}G + request {c.get('gpu_gb')}G > cap {policy['gpu_cap_gb']}G",
                        s, policy, 5)

    ev = {"event": "admit", "run_id": args.run_id, "cls": cls, "pid": os.getppid(),
          "ts": s["ts"], "_ts_epoch": time.time(), "reason": args.reason or "",
          "snapshot": s}
    append(ev)
    print(json.dumps({"verdict": "ADMIT", "run_id": args.run_id, "cls": cls,
                      "snapshot_summary": {k: s[k] for k in
                          ("free_pct", "swap_used_gb", "compressed_gb", "ssd_free_gb", "procs")}},
                     ensure_ascii=False, indent=1))
    return 0


def deny(kind, why, s, policy, code):
    ev = {"event": "deny", "kind": kind, "why": why, "cls": None,
          "ts": s["ts"], "_ts_epoch": time.time()}
    print(json.dumps({"verdict": "DENY", "kind": kind, "why": why,
                      "recover_wait_s": policy["recover_wait_s"],
                      "snapshot_summary": {k: s.get(k) for k in
                          ("free_pct", "swap_used_gb", "compressed_gb", "procs")}},
                     ensure_ascii=False, indent=1))
    return code


def release(args):
    ev = {"event": "release", "run_id": args.run_id, "rc": args.rc,
          "artifact": args.artifact, "ts": now_iso(), "_ts_epoch": time.time()}
    append(ev)
    print(json.dumps({"released": args.run_id, "rc": args.rc}, ensure_ascii=False))
    return 0


def status(_args):
    policy = load_policy()
    s = snapshot(policy)
    live = read_registry()
    out = {"snapshot": s, "resident_gpu_est": resident_gpu_estimate(s, policy),
           "swap_rate_gb_min": swap_rate(policy),
           "live_registrations": {rid: {"cls": r["cls"], "pid": r["pid"], "ts": r["ts"]}
                                  for rid, r in live.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def watch(args):
    policy = load_policy()
    prev = None
    alerts = 0
    try:
        while True:
            s = snapshot(policy)
            s["_epoch"] = time.time()
            with open(MON, "a") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            alert = None
            if s["free_pct"] is not None and s["free_pct"] < 15:
                alert = f"free_pct={s['free_pct']}"
            elif (s.get("swap_used_gb") or 0) > policy["swap_used_warn_gb"]:
                rate = swap_rate(policy)
                if rate is not None and rate > policy["swap_rate_block_gb_per_min"]:
                    alert = f"swap {s['swap_used_gb']}G growing {rate}G/min"
            if alert and alerts < 20:
                print(f"ALERT {s['ts']} {alert}", flush=True)
                alerts += 1
            prev = s
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=status)
    r = sub.add_parser("request")
    r.add_argument("cls", choices=list(DEFAULT_POLICY["classes"]))
    r.add_argument("--run-id", required=True)
    r.add_argument("--reason", default="")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=request)
    rl = sub.add_parser("release")
    rl.add_argument("--run-id", required=True)
    rl.add_argument("--rc", type=int, default=None)
    rl.add_argument("--artifact", default=None)
    rl.set_defaults(fn=release)
    w = sub.add_parser("watch")
    w.add_argument("--interval", type=int, default=120)
    w.set_defaults(fn=watch)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
