#!/bin/bash
# P3-C D arm v2: same as e0j but on model_native_v2 (token_refiner qkv spliced to original bytes).
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
D=$EV/e0k_native_v2_20261126
mkdir -p "$D"
cd "$REPO"
echo "=== e0k native model_native_v2 start $(date) ===" > "$D/run_log.txt"
df -m /Volumes/SSD2T | tail -1 >> "$D/run_log.txt"
env H3_NATIVE_INT8=1 H3_COND_DEBUG=1 H3_DUMP_DIR="$D" H3_DUMP_ALL=1 H3_DUMP_QUOTA_GB=80 \
  /usr/bin/time -l ./h3 \
  -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 2 --seed 1 \
  --profile -o "$D/dump_run.mp4" >> "$D/run_log.txt" 2>&1
RC=$?
echo "=== exit=$RC $(date) ===" >> "$D/run_log.txt"
echo "E0K_NATIVE_V2_DONE rc=$RC taps=$(cat "$D/manifest.txt" 2>/dev/null | grep -c '')"
