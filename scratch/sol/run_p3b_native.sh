#!/bin/bash
# P3-B: first clean native A8W8 run (H3_NATIVE_INT8=1), 640x352x25f steps2 seed1.
# Clean = no dump/instrumentation env vars. Sampler logs swap/pressure/RSS every 20s.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
OUTDIR=/Users/cyliu99/AI/MV_projects/h3c_native
mkdir -p "$OUTDIR"
LOG="$OUTDIR/p3b_native_640x352x25f.log"
SAMP="$OUTDIR/p3b_sampler.log"
MP4="$OUTDIR/p3b_native_640x352x25f.mp4"
cd "$REPO"

( while true; do
    ts=$(date +%H:%M:%S)
    sw=$(sysctl -n vm.swapusage)
    pr=$(memory_pressure -Q 2>/dev/null | grep -o 'System-wide memory free percentage: [0-9]*%')
    rss=$(ps -axo rss=,comm= | awk '/strata|h3$/{printf "%s:%.1fG ",FILENAME,$1/1048576}' FS= )
    rss=$(ps -axo rss=,comm= | awk '$2 ~ /\/strata|h3$/ {printf "%s=%.1fG ", $2, $1/1048576}')
    echo "$ts | $sw | $pr | $rss" >> "$SAMP"
    sleep 20
  done ) &
SAMPID=$!
trap 'kill $SAMPID 2>/dev/null' EXIT

echo "=== P3-B native run start $(date) ===" > "$LOG"
H3_NATIVE_INT8=1 /usr/bin/time -l ./h3 \
  -d /Users/cyliu99/AI/h3c_mac_20261125/model_native \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 25 --steps 2 --seed 1 \
  --profile -o "$MP4" >> "$LOG" 2>&1
RC=$?
echo "=== exit=$RC $(date) ===" >> "$LOG"
ls -l "$MP4" >> "$LOG" 2>&1
echo "P3B_DONE rc=$RC"
