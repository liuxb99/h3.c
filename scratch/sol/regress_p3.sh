#!/bin/bash
# P3 permanent regression gate (user rule 2026-11-26):
# a run counts as success ONLY IF
#   1. the tiny native generation exits 0 AND writes an mp4,
#   2. every compute tap dump is finite (regress_taps.py: NaN/Inf/empty = FAIL,
#      which catches kernels that never dispatched or never wrote output),
#   3. the refine condition GEMM took a real dispatch path (LINEARDBG present)
#      and its output audit has zero NaN (CONDDBG out nan=0),
#   4. no "Metal command failed" / MPS import failure in the run log.
# Usage: regress_p3.sh [dump_dir]   (default: scratch/regress_p3_run)
set -u
ROOT=/Users/cyliu99/AI/h3c_mac_20261125/h3c
D=${1:-$ROOT/scratch/regress_p3_run}
MODEL=${P3_MODEL:-/Users/cyliu99/AI/h3c_mac_20261125/model_native_v2}
rm -rf "$D"; mkdir -p "$D"
cd "$ROOT"
env H3_NATIVE_INT8=1 H3_COND_DEBUG=1 H3_DUMP_DIR="$D" H3_DUMP_ALL=1 \
  ./h3 -d "$MODEL" -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 64 --height 64 --frames 25 --steps 2 --seed 1 -o "$D/tiny.mp4" \
  > "$D/run_log.txt" 2>&1
rc=$?
fail=0
[ $rc -ne 0 ] && { echo "FAIL: engine exit $rc"; fail=1; }
[ -s "$D/tiny.mp4" ] || { echo "FAIL: no mp4 output"; fail=1; }
grep -q "Metal command failed" "$D/run_log.txt" && { echo "FAIL: Metal command error in log"; fail=1; }
grep -q "LINEARDBG path=" "$D/run_log.txt" || { echo "FAIL: no LINEARDBG dispatch evidence"; fail=1; }
grep "CONDDBG" "$D/run_log.txt" | grep -v "nan=0" | grep -q " out " && { echo "FAIL: CONDDBG out contains NaN"; fail=1; }
uv run --with numpy -q python "$ROOT/scratch/sol/regress_taps.py" "$D" || fail=1
# evidence hygiene: keep dumps small, prune raws after verdict unless KEEP_DUMPS=1
[ "${KEEP_DUMPS:-0}" = 1 ] || rm -f "$D"/*.raw
if [ $fail -eq 0 ]; then echo "REGRESS_P3_PASS"; else echo "REGRESS_P3_FAIL"; fi
exit $fail
