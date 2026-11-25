#!/bin/zsh
cd /Users/cyliu99/AI/h3c_mac_20261125/h3c || exit 1
mkdir -p scratch/sol scratch/exact
MODEL=/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3
PROMPT='a red fox running through a green meadow on a sunny day'
COMMON=(--width 640 --height 352 --frames 13 --steps 6 --seed 20261125 --profile)
# A: Sol-Attn tau=1.3 (bf16 row-major attention output so the Sol branch runs)
env H3_SOL_ATTN=1.3 ./h3 -d $MODEL -p $PROMPT -o scratch/sol/video.mp4 \
    --use-slower-row-major-attention-output $COMMON > scratch/sol/run.log 2>&1
echo "sol rc=$?" >> scratch/sol/run.log
# B: exact attention, int8 head-major output (8.837s baseline recipe)
./h3 -d $MODEL -p $PROMPT -o scratch/exact/video.mp4 $COMMON \
    > scratch/exact/run.log 2>&1
echo "exact rc=$?" >> scratch/exact/run.log
grep -E "Euler denoise" scratch/sol/run.log scratch/exact/run.log
grep -E "rc=" scratch/sol/run.log scratch/exact/run.log
ls -l scratch/sol/video.mp4 scratch/exact/video.mp4 2>/dev/null
echo AB_DONE
