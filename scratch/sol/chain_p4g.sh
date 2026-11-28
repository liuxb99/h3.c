#!/bin/bash
# P4G: post-patch semantic re-run, 3 prompts, fast contract 256x256x22f steps2 seed1 (same as P4E), native arm v2.
# GPU-serial: waits for P4F CHAIN_P4F_DONE, then ComfyX-queue idle poll (<=2h/round), then 3 runs.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
P4LOG=$REPO/scratch/sol/chain_p4f.log
LOG=$REPO/scratch/sol/chain_p4g.log
cd "$REPO" || exit 9

idle () {
  local q
  q=$(curl -s -m3 http://127.0.0.1:8390/queue 2>/dev/null) || return 0
  [ -z "$q" ] && return 0
  echo "$q" | python3 -c 'import json,sys
d=json.load(sys.stdin)
sys.exit(0 if not d.get("queue_running") and not d.get("queue_pending") else 1)' 2>/dev/null
}

run_prompt () {
  local tag=$1 p="$2"
  local D=$EV/p4g_$tag; mkdir -p "$D"
  env H3_DUMP_DIR="$D" H3_DUMP_STEPS=0,1 H3_DUMP_TAPS="b0_s*,e1_after*" \
    ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 \
    -p "$p" --width 256 --height 256 --frames 22 --steps 2 --seed 1 \
    -o "$D/out.mp4" > "$D/run_log.txt" 2>&1
  local RC=$?
  echo "PROMPT_DONE tag=p4g_$tag rc=$RC mp4_bytes=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0)"
}

{
echo "=== P4G start $(date) ==="
for i in $(seq 1 480); do
  grep -q "CHAIN_P4F_DONE" "$P4LOG" && break
  sleep 30
done
grep -q "CHAIN_P4F_DONE" "$P4LOG" || { echo "P4F_TIMEOUT — abort"; exit 7; }
[ -s "$EV/p4f_nativepost640/out.mp4" ] || echo "WARN: p4f mp4 missing (check p4f log)"
for round in 1 2 3; do
  for i in $(seq 1 240); do
    pgrep -f "./h3 -d" >/dev/null && { sleep 30; continue; }
    idle && break
    sleep 30
  done
  if ! idle; then echo "GPU_BUSY_TIMEOUT round=$round"; continue; fi
  case $round in
    1) run_prompt p1_cat "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格";;
    2) run_prompt p2_corgi_piano "A corgi playing a tiny piano on a stage, bright cartoon style";;
    3) run_prompt p3_ferris "一個女孩坐在游樂園的摩天輪上，夕陽，動漫風格";;
  esac
done
echo "CHAIN_P4G_DONE $(date)"
} >> "$LOG" 2>&1
