#!/bin/bash
# M1：resident int8（row-fc2 M5 快路徑），小尺寸，與 Strata 共存試點
set -uo pipefail
H3DIR=/Users/cyliu99/AI/h3c_mac_20261125/h3c
MODEL=/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3
OUT=/Users/cyliu99/AI/h3c_mac_20261125/runs
cd "$H3DIR"
/usr/bin/time -l ./h3 -d "$MODEL" \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 17 --steps 10 --seed 42 \
  --use-int8-row-fc2 --profile \
  -o "$OUT/m1_int8.mp4" 2>&1 | tee "$OUT/m1_int8.log"
