#!/bin/bash
# P4E: 3-semantic-prompt ablation at native-INT8 tiny contract (fast), with refiner subtaps.
# Prompts: P1 action-only(corgi+piano, no style), P2 text-render positive control (sign "OPEN 24 HOURS"), P3 zh scene.
# If P2 renders the sign text correctly but P1/P3 scenes detach => text path alive, scene conditioning broken.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
cd "$REPO" || exit 9
pgrep -f "./h3 -d" && { echo GPU_BUSY; exit 8; }
run() { TAG=$1; P=$2; D=$EV/p4e_$TAG; mkdir -p "$D"
  echo "=== $TAG $(date) ==="; df -m /Volumes/SSD2T | tail -1
  env H3_NATIVE_INT8=1 H3_REFINE_SUBTAP=1 H3_DUMP_DIR="$D" H3_DUMP_STEPS=0 H3_DUMP_TAPS="rb*,e*,step0_*" \
    ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 -p "$P" --width 256 --height 256 --frames 22 --steps 2 --seed 1 -o "$D/out.mp4" > "$D/run_log.txt" 2>&1
  RC=$?; echo "DONE tag=$TAG rc=$RC raw=$(ls "$D"/*.raw 2>/dev/null | wc -l) mp4=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0)"; }
{
curl -s -m3 http://127.0.0.1:8390/queue | python3 -c 'import json,sys;q=json.load(sys.stdin);sys.exit(0 if not q["queue_running"] and not q["queue_pending"] else 1)' || { echo COMFYX_BUSY; exit 8; }
run p1_corgi "A corgi plays grand piano on a stage, spotlight"
run p2_sign  'A shop front at dusk with a glowing neon sign reading "OPEN 24 HOURS", wet pavement reflections'
run p3_teahouse "夕陽下的江南水鄉茶座，老船夫撐船經過石橋，水墨畫風格"
echo "CHAIN_P4E_DONE $(date)"
} >> scratch/sol/chain_p4e.log 2>&1
