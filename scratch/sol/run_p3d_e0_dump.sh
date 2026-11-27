#!/bin/bash
# P3-D: dequant arm with e0 (encode_forward) taps — shared patch/cond chain is
# arm-independent; one arm suffices. Same recipe as baseline.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Users/cyliu99/AI/h3c_mac_20261125/evidence
D=$EV/e0_dump_20261126
mkdir -p "$D"
cd "$REPO"
echo "=== e0 dump run start $(date) ===" > "$D/run_log.txt"
H3_DUMP_DIR="$D" H3_DUMP_ALL=1 /usr/bin/time -l ./h3 \
  -d /Users/cyliu99/AI/h3c_mac_20261125/model_native \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 2 --seed 1 \
  --profile -o "$D/dump_run.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
echo "=== exit=$RC $(date) ===" >> "$D/run_log.txt"
echo "P3D_E0_DUMP_DONE rc=$RC taps=$(ls "$D"/*.raw 2>/dev/null | wc -l | tr -d ' ')"
