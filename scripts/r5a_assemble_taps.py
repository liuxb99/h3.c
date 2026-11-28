#!/usr/bin/env python
"""Assemble R5a taps.json from scratch/sol/r5a_phase_*.json + gate/sd-cli records."""
import json, os, sys, subprocess, datetime

REPO = "/Users/cyliu99/AI/h3c_mac_20261125/h3c"
SOL = os.path.join(REPO, "scratch/sol")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs/evidence/r5a_adaln_taps_20261128")

def load(p):
    fp = os.path.join(SOL, p)
    return json.load(open(fp)) if os.path.exists(fp) else None

def shell(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return "ERR " + str(e)

taps = {}
t1 = load("r5a_phase_t1.json")
if t1: taps["t1_gguf_table_lerp_vs_contract"] = t1
t2 = load("r5a_phase_t2.json")
if t2: taps["t2_perblock_row_order_vs_contract"] = t2
t3 = load("r5a_phase_t3.json")
if t3: taps["t3_weight_level_perindex_3blocks"] = t3
t4 = load("r5a_phase_t4.json")
if t4: taps["t4_perstep_table_vs_cpu_forward"] = t4

pgrep = shell("pgrep -x sd-cli || echo EMPTY")
swap = shell("sysctl -n vm.swapusage")
mp = shell("memory_pressure -Q 2>/dev/null | tail -2")
taps["sd_cli_forward_tap"] = {
    "status": "DEFERRED_coordination_window",
    "gate_measurements": {"pgrep_sd_cli": pgrep, "vm.swapusage": swap,
                          "memory_pressure": mp,
                          "captured_at": datetime.datetime.now().isoformat(timespec="seconds")},
    "rationale": "task rule: run sd-cli 1-step only if pgrep empty AND memory idle AND swap<10G. "
                 "machine under sustained pressure from sibling parallel task (compressor > gate cap; "
                 "admit.py DENIED light_cpu repeatedly 19:20-19:50), so memory not idle -> DEFERRED, "
                 "watchdog-panic red line not risked for an auxiliary tap.",
}
taps["admission_gate_log"] = load("r5a_admission_log.json") or {}
taps["recovered_prior_artifacts"] = {
    "used": ["scratch/r5a_gguf_tensor_infos.json (GGUF tensor table, re-verified against live GGUF header in t1)",
             "scratch/sol/engine_backup_prepatch/ (559 files, no adaln weights - not needed)",
             "scratch/sol/r5a_gguf_tensor_infos.json (same tensor table copy)"],
    "note": "killed R5a subagent left no evidence dir; tensor-info pre-scan was reusable and independently re-verified from the GGUF file itself before use",
}

layout = None
if t2:
    layout = {
        "final_conclusion": t2.get("conclusion_layout"),
        "evidence": "t2 correlation fingerprint + t3 coefficient per-index identity (see per-tap entries)",
    }
taps["layout_96768_conclusion"] = layout

verdicts = {k: taps[k].get("verdict", "NOT_RUN") for k in taps if k.startswith("t") and isinstance(taps[k], dict)}
taps["verdicts_summary"] = verdicts

os.makedirs(OUT, exist_ok=True)
json.dump(taps, open(os.path.join(OUT, "taps.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "taps.json"), "keys:", list(taps))
