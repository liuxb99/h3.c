#!/bin/bash
# P3-C: arm A (original BF16 Golden weights) at the 2790-token recipe
# (640x352x25f, steps 2, seed 1) so it is shape-comparable with e0d/e0f/native dumps.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
D=$EV/a2790_dump_baseline
mkdir -p "$D"
cd "$REPO"
echo "=== a2790 dump run start $(date) ===" > "$D/run_log.txt"
H3_DUMP_DIR="$D" H3_DUMP_ALL=1 /usr/bin/time -l ./h3 \
  -d /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3 \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 2 --seed 1 \
  --profile -o "$D/dump_run.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
echo "=== exit=$RC $(date) ===" >> "$D/run_log.txt"
echo "P3D_A2790_DUMP_DONE rc=$RC taps=$(ls "$D"/*.raw 2>/dev/null | wc -l | tr -d ' ')"
