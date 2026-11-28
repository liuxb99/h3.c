#!/bin/bash
# Fail-closed engine load gate. Usage: verify_ready.sh <asset_path> [manifest_dir]
# Blocks engine/model loading unless BOTH:
#   1) a <asset_path>.ready marker exists (written by the copy/installer ONLY after
#      full copy + fsync + full-sha verification), and
#   2) asset_audit --single passes (size + full sha + read_only + header_contract +
#      layout-contract inode rules).
# Exit: 0 LOAD_READY, 2 LOAD_BLOCKED (violation), 3 usage error.
set -u
[ $# -ge 1 ] || { echo "usage: verify_ready.sh <asset_path> [manifest_dir]"; exit 3; }
P="$1"
MD="${2:-$(cd "$(dirname "$0")/.." && pwd)/model_manifests}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$P.ready" ]; then
  echo "LOAD_BLOCKED: missing ready marker ($P.ready) — asset may be mid-copy"
  exit 2
fi
OUT=$(python3 "$HERE/asset_audit.py" --manifest-dir "$MD" --single "$P" 2>&1)
RC=$?
echo "$OUT" | tail -8
if [ $RC -ne 0 ]; then
  echo "LOAD_BLOCKED: asset_audit --single rc=$RC"
  exit 2
fi
echo "LOAD_READY: $P"
exit 0
