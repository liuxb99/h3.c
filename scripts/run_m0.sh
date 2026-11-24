#!/bin/bash
# M0 端到端基線：FL2VA 小解析度 T2V（ssd-streaming 共存模式，護欄：MemAvailable 檢查）
set -euo pipefail
H3DIR=/Users/cyliu99/AI/h3c_mac_20261125/h3c
MODEL=/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3
OUT=/Users/cyliu99/AI/h3c_mac_20261125/runs
mkdir -p "$OUT"
# 護欄：可用記憶體 < 30G 拒跑
FREE_GB=$(memory_pressure | awk '/free percentage/ {print $NF}' | sed 's/%//')
echo "memory free %: $FREE_GB"
# Strata 探針
curl -s --max-time 5 http://127.0.0.1:8080/health >/dev/null && echo "strata health ok" || echo "strata probe failed"
cd "$H3DIR"
/usr/bin/time -l ./h3 -d "$MODEL" \
  -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 640 --height 352 --frames 17 --steps 10 --seed 42 \
  --ssd-streaming --profile \
  -o "$OUT/m0_t2v.mp4" 2>&1 | tee "$OUT/m0_t2v.log"
