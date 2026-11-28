#!/bin/bash
# P4C: block0 internals dump at 640 s2 (post tap-name fix) to pinpoint first divergence inside attention.
# Run only when GPU idle (guard inside). Eager first; native only if eager taps look clean up to attention input.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
PROMPT="一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格"
cd "$REPO" || exit 9
{
echo "== CHAIN_P4C $(date) =="
pgrep -f "./h3 -d" && { echo GPU_BUSY; exit 8; }
pgrep -f "main.py --port 8390" >/dev/null && curl -s -m3 http://127.0.0.1:8390/queue | grep -q '"running": *null' || { sleep 120; curl -s -m3 http://127.0.0.1:8390/queue | grep -q '"running": *null' || { echo GPU_BUSY_COMFYX; exit 8; }; }
D=$EV/p4c_eager640_s2_b0; mkdir -p "$D"
df -m /Volumes/SSD2T | tail -1
env H3_DUMP_DIR="$D" H3_DUMP_STEPS=0,1 H3_DUMP_BLOCKS=0,1 H3_DUMP_TAPS="b0_*,b1_*,e*,step0_*" \
  ./h3 -d /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3 -p "$PROMPT" --width 640 --height 352 --frames 25 --steps 2 --seed 1 -o "$D/out.mp4" > "$D/run_log.txt" 2>&1
echo "ARM_DONE eager_s2 rc=$? files=$(ls "$D"/*.raw 2>/dev/null | wc -l)"
echo "CHAIN_P4C_DONE $(date)"
} >> scratch/sol/chain_p4c.log 2>&1
