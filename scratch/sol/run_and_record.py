#!/usr/bin/env python3
"""run_and_record.py TAG -- cmd... : run one chain step, record rc/dur into
pipeline_status.json (same schema as finish_chain), never breaks a sequence."""
import datetime
import json
import os
import subprocess
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(S, "pipeline_status.json")
REPO = os.path.dirname(os.path.dirname(S))


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load():
    with open(STATUS) as fh:
        return json.load(fh)


def record(tag, rc, dur):
    d = load()
    d.setdefault(tag, {}).update({"rc": rc, "t": now(), "dur_s": dur})
    d["updated"] = now()
    with open(STATUS, "w") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main():
    tag = sys.argv[1]
    cmd = sys.argv[3:] if sys.argv[2] == "--" else sys.argv[2:]
    log = open(os.path.join(S, f"chain_{tag}.log"), "a")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=log,
                          timeout=7200, stdin=subprocess.DEVNULL)
    dur = round(time.time() - t0, 1)
    log.close()
    record(tag, proc.returncode, dur)
    print(f"{tag} rc={proc.returncode} dur={dur}s")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
