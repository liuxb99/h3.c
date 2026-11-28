#!/usr/bin/env python3
"""Revert official convrot file to pre-patch bytes using engine_backup_prepatch manifest.
Writes reverted copy to <out>, verifies sha256 == official LFS oid, prints result. Does NOT
replace the live file itself (caller does atomic mv)."""
import json, hashlib, os, sys

BACKUP = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/engine_backup_prepatch"
CURRENT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
OFFICIAL_SHA = "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/.revert_tmp.safetensors"

man = json.load(open(os.path.join(BACKUP, "manifest.json")))
ents = man["entries"]
assert man.get("applied"), "manifest not marked applied"

# copy current file first (streaming), then overwrite damaged regions from .old.bin
import shutil
print(f"copying current file -> {OUT}", flush=True)
shutil.copyfile(CURRENT, OUT)

def slug(name): return name.replace(".", "_")

with open(OUT, "r+b") as f:
    for i, e in enumerate(ents):
        ob = os.path.join(BACKUP, slug(e["name"]) + ".old.bin")
        data = open(ob, "rb").read()
        assert len(data) == e["len"], f"{e['name']}: old.bin len {len(data)} != manifest len {e['len']}"
        h8 = hashlib.sha256(data).hexdigest()[:16]
        assert h8 == e["sha_before"], f"{e['name']}: old.bin sha {h8} != manifest sha_before {e['sha_before']}"
        f.seek(e["offset"]); f.write(data)
        if (i + 1) % 50 == 0: print(f"restored {i+1}/{len(ents)}", flush=True)
print(f"restored {len(ents)}/{len(ents)} regions", flush=True)

# verify full-file sha + spot-check: recompute per-region sha_before by reading OUT back
h = hashlib.sha256()
with open(OUT, "rb") as f:
    while True:
        b = f.read(1 << 24)
        if not b: break
        h.update(b)
full = h.hexdigest()
print(f"REVERT full sha256: {full}")
print(f"OFFICIAL expected : {OFFICIAL_SHA}")
print("REVERT_SHA_OK" if full == OFFICIAL_SHA else "REVERT_SHA_MISMATCH")
sys.exit(0 if full == OFFICIAL_SHA else 1)
