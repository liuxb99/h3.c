#!/usr/bin/env python3
"""finalize_strip_manifest.py — after the strip-renamed built asset passes a
full gate rerun, fold the rename + new verdict into build_manifest.json.

Usage: finalize_strip_manifest.py <gate_json> <gate_log> <gate_exit>
Steps: assert gate verdict -> stream sha256 of the 34G file -> rewrite
manifest (atomic tmp+rename), keeping every pre-rename field, adding
sidecar_rename + gate_rerun4_strip blocks. Payload/tensor SHAs stay valid
(rename preserved all data_offsets; documented in the manifest itself)."""
import json, hashlib, os, sys, datetime

GATE_JSON, GATE_LOG, GATE_EXIT = sys.argv[1], sys.argv[2], int(sys.argv[3])
MODEL_DIR = "/Volumes/SSD2T/models/MiniMax-H3-h3c-built"
SINGLE = os.path.join(MODEL_DIR, "minimax_h3_ref2va_int8_convrot_h3c.safetensors")
MANIFEST = os.path.join(MODEL_DIR, "build_manifest.json")
BAK = "/Users/cyliu99/AI/h3c_mac_20261125/evidence/convrot_audit_20261128/built_header_before.json"

def sha256_stream(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_small(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

g = json.load(open(GATE_JSON))
failures = g.get("failures", [])
summary = g.get("summary", {})
if failures or GATE_EXIT != 0:
    print(f"FAIL: gate_exit={GATE_EXIT} failures={len(failures)}")
    for f in failures[:10]:
        print(" ", str(f)[:200])
    sys.exit(2)

m = json.load(open(MANIFEST))
new_sha = sha256_stream(SINGLE)
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

m["pre_strip_rename_sha256"] = m["sha256"]
m["sha256"] = new_sha
m["sidecar_form"] = "strip (X.weight -> X.weight_scale / X.comfy_quant), " \
                    "production loader convention measured on official " \
                    "comfy-int8 AND h3c-engine asset (sidecar_keys_probe.py)"
m["sidecar_rename"] = {
    "renamed_utc": now,
    "script": "h3c/scratch/sol/rename_built_sidecar_strip.py",
    "keys_renamed": 500,
    "mechanism": "header-JSON-only rewrite; new JSON space-padded to exact "
                 "old length; u64 prefix, data start offset and all payload "
                 "bytes byte-identical (per-tensor data_offsets asserted "
                 "against pre-rename header backup)",
    "pre_rename_header_sha256": sha256_small(BAK),
    "pre_rename_header_backup": BAK,
}
small = {k: v for k, v in summary.items()
         if not isinstance(v, (list, dict)) or len(str(v)) < 300}
m["gate_rerun4_strip"] = {
    "run": now, "gate_exit": GATE_EXIT, "verdict": "PASS",
    "summary_small": small,
    "candidate": summary.get("candidate"),
    "evidence_shas256": {"gate_json": sha256_small(GATE_JSON),
                         "gate_log": sha256_small(GATE_LOG)},
}
m["ready_utc"] = now

tmp = MANIFEST + ".tmp"
json.dump(m, open(tmp, "w"), indent=2, ensure_ascii=False)
os.replace(tmp, MANIFEST)
print(f"FINALIZE OK new_sha256={new_sha}")
print(f"pre_strip_sha={m['pre_strip_rename_sha256']}")
