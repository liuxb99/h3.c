#!/bin/bash
# P3-F: same-fixed-binary A/D pair at production 640x352x25f (a2790 prompt/seed).
# usage: run_p3f_pair.sh TAG MODELDIR [extra env assignments...]
set -u
TAG=$1; MODEL=$2; shift 2
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
D=$EV/$TAG
mkdir -p "$D"
cd "$REPO"
echo "=== $TAG start $(date) ===" > "$D/run_log.txt"
df -m /Volumes/SSD2T | tail -1 >> "$D/run_log.txt"
env "$@" H3_DUMP_DIR="$D" H3_DUMP_ALL=1 H3_DUMP_QUOTA_GB=60 \
  /usr/bin/time -l ./h3 -d "$MODEL" \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width ${W:-640} --height ${H:-352} --frames ${F:-25} --steps 2 --seed 1 \
  -o "$D/out.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
grep "maximum resident" "$D/run_log.txt"
echo "P3F_${TAG}_DONE rc=$RC taps=$(ls "$D"/*.raw 2>/dev/null | wc -l | tr -d ' ') mp4=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0)"
