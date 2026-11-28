#!/bin/bash
# P3-C decisive: native INT8 arm on COMFY FLAT dir (loader mode that E0F
# proved byte-correct). Env scrubbed explicitly on both sides.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
D=$EV/e0h_native_comfyflat_20261126
mkdir -p "$D"
cd "$REPO"
echo "=== e0h native+comfyflat start $(date) ===" > "$D/run_log.txt"
env H3_NATIVE_INT8=1 H3_DUMP_DIR="$D" H3_DUMP_ALL=1 /usr/bin/time -l ./h3 \
  -d /Volumes/SSD2T/models/MiniMax-H3-comfy-int8 \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 2 --seed 1 \
  --profile -o "$D/dump_run.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
echo "=== exit=$RC $(date) ===" >> "$D/run_log.txt"
echo "P3C_E0H_DONE rc=$RC taps=$(ls "$D"/*.raw 2>/dev/null | wc -l | tr -d ' ')"
