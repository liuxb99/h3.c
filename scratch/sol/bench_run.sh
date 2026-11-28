#!/bin/bash
# bench_run.sh TAG MODELDIR OUTDIR W H FRAMES STEPS [extra h3 args...]
# Runs one h3 generation under measurement: wall clock, max RSS, peak swap delta.
# Writes $OUTDIR/run_log.txt, $OUTDIR/metrics.txt
set -u
TAG=$1; MODEL=$2; OUT=$3; W=$4; H=$5; F=$6; S=$7; shift 7
mkdir -p "$OUT"
PROMPT="一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格"
( while :; do sysctl -n vm.swapusage; sleep 2; done > "$OUT/swap_samples.txt" 2>&1 & echo $! > "$OUT/swap_sampler.pid" )
/usr/bin/time -l env "$@" ./h3 -d "$MODEL" -p "$PROMPT" --width "$W" --height "$H" --frames "$F" --steps "$S" -o "$OUT/out.mp4" > "$OUT/run_log.txt" 2>&1
rc=$?
kill "$(cat "$OUT/swap_sampler.pid")" 2>/dev/null
swappeak=$(awk '{gsub(/[a-zA-Z()=]/," ");gsub(/M/,""); if($4+0>mx)mx=$4+0} END{printf "%.0f", mx}' "$OUT/swap_samples.txt")
maxrss=$(grep "maximum resident" "$OUT/run_log.txt" | awk '{print $1}')
wall=$(grep " real " "$OUT/run_log.txt" | awk '{print $1}')
{ echo "tag=$TAG rc=$rc wall_s=${wall:-NA} maxrss_bytes=${maxrss:-NA} swap_peak_M=$swappeak out=$([ -s "$OUT/out.mp4" ] && stat -f%z "$OUT/out.mp4" || echo 0)"; } > "$OUT/metrics.txt"
cat "$OUT/metrics.txt"
exit $rc
