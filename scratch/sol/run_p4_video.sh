#!/bin/bash
# P4: golden-recipe VIDEO acceptance at 640x352x25f, steps 20, seed 1 (no tap dumps -> speed).
# usage: run_p4_video.sh TAG MODELDIR [extra env assignments...]
set -u
TAG=$1; MODEL=$2; shift 2
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
D=$EV/$TAG
mkdir -p "$D"
cd "$REPO"
echo "=== $TAG start $(date) ===" > "$D/run_log.txt"
df -m /Volumes/SSD2T | tail -1 >> "$D/run_log.txt"
env "$@" \
  /usr/bin/time -l ./h3 -d "$MODEL" \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 20 --seed 1 \
  -o "$D/out.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
grep -E "maximum resident|real" "$D/run_log.txt"
echo "P4_${TAG}_DONE rc=$RC mp4=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0) step20"
