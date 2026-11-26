#!/bin/bash
# P3-e0m: sentinel + branch-selection probe on the refine condition GEMM.
EV=/Volumes/SSD2T/h3c_evidence_20261126
mkdir -p $EV/e0m_sentinel
cd /Users/cyliu99/AI/h3c_mac_20261125/h3c
env H3_NATIVE_INT8=1 H3_COND_DEBUG=1 H3_DUMP_DIR=$EV/e0m_sentinel H3_DUMP_ALL=1 \
  ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
  --width 64 --height 64 --frames 25 --steps 2 --seed 1 -o $EV/e0m_sentinel/tiny.mp4 > $EV/e0m_sentinel/run_log.txt 2>&1
echo "e0m rc=$? taps=$(cat $EV/e0m_sentinel/manifest.txt 2>/dev/null|grep -c '')"
