#!/bin/zsh
cd /Users/cyliu99/AI/h3c_mac_20261125/h3c || exit 1
env H3_SOL_ATTN=-1000 ./h3 -d /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3 \
  -p 'a red fox running through a green meadow on a sunny day' \
  -o scratch/sol/degen.mp4 --use-slower-row-major-attention-output \
  --width 640 --height 352 --frames 13 --steps 6 --seed 20261125 --profile \
  > scratch/sol/degen.log 2>&1
echo "degen rc=$?" >> scratch/sol/degen.log
grep -aE "Euler denoise|wrote" scratch/sol/degen.log
echo DEGEN_DONE
