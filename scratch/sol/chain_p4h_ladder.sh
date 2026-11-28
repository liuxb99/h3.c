#!/bin/bash
# P4H: resolution ladder per user spec — 544x288x124f steps6 (3 prompts), then 1280x720x121f steps6 (3 prompts).
# GPU-serial: waits for CHAIN_P4G_DONE, then ComfyX-queue idle before EACH run (never interrupt production).
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
P4GLOG=$REPO/scratch/sol/chain_p4g.log
LOG=$REPO/scratch/sol/chain_p4h.log
cd "$REPO" || exit 9

idle () {
  local q
  q=$(curl -s -m3 http://127.0.0.1:8390/queue 2>/dev/null) || return 0
  [ -z "$q" ] && return 0
  echo "$q" | python3 -c 'import json,sys
d=json.load(sys.stdin)
sys.exit(0 if not d.get("queue_running") and not d.get("queue_pending") else 1)' 2>/dev/null
}

run_job () {
  local tag=$1 p=$2 w=$3 h=$4 f=$5
  local D=$EV/$tag; mkdir -p "$D"
  for i in $(seq 1 480); do idle && break; sleep 30; done
  idle || { echo "SKIP_BUSY tag=$tag"; return 8; }
  env H3_DUMP_DIR="$D" H3_DUMP_STEPS=0 H3_DUMP_TAPS="b0_s*,e1_after*" \
    /usr/bin/time -l ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 \
    -p "$p" --width "$w" --height "$h" --frames "$f" --steps 6 --seed 1 \
    -o "$D/out.mp4" > "$D/run_log.txt" 2>&1
  local RC=$?
  echo "ARM_DONE tag=$tag rc=$RC mp4_bytes=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0) $(grep -E 'maximum resident|real' "$D/run_log.txt" | tr '\n' ' ')"
}

{
echo "=== P4H ladder start $(date) ==="
for i in $(seq 1 1440); do grep -q "CHAIN_P4G_DONE" "$P4GLOG" && break; sleep 30; done
grep -q "CHAIN_P4G_DONE" "$P4GLOG" || { echo "P4G_TIMEOUT — abort"; exit 7; }
run_job p4h_544_cat "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" 544 288 124
run_job p4h_544_corgi "一隻柯基犬坐在鋼琴前歡快地用爪子彈琴，音樂廳舞台，卡通風格" 544 288 124
run_job p4h_544_girl "一個小女孩坐在摩天輪上開心地笑，夕陽遊樂園，卡通風格" 544 288 124
run_job p4h_1280_cat "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" 1280 720 121
run_job p4h_1280_corgi "一隻柯基犬坐在鋼琴前歡快地用爪子彈琴，音樂廳舞台，卡通風格" 1280 720 121
run_job p4h_1280_girl "一個小女孩坐在摩天輪上開心地笑，夕陽遊樂園，卡通風格" 1280 720 121
echo "CHAIN_P4H_DONE $(date)"
} >> "$LOG" 2>&1
