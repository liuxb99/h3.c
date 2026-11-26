#!/bin/bash
EV=/Volumes/SSD2T/h3c_evidence_20261126
run(){ arm=$1; renv=$2; mkdir -p $EV/e0l_$arm; cd /Users/cyliu99/AI/h3c_mac_20261125/h3c;
 env H3_NATIVE_INT8=1 H3_REUSE_MPS_COMMAND=$renv H3_COND_DEBUG=1 H3_DUMP_DIR=$EV/e0l_$arm H3_DUMP_ALL=1 \
 ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 -p "一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格" \
 --width 64 --height 64 --frames 25 --steps 2 --seed 1 -o $EV/e0l_$arm/tiny.mp4 > $EV/e0l_$arm/run_log.txt 2>&1;
 echo "$arm rc=$? taps=$(cat $EV/e0l_$arm/manifest.txt 2>/dev/null|grep -c '')"; }
run reuse0 0; run reuse1 1; echo E0L_AB_DONE
