#!/bin/bash
# P4B: golden-validity rerun at 640x352x25f steps20 seed1 (user-approved plan 1, 2026-11-27).
# Unified shader axis: MiniMaxAI dir shader now symlinks repo h3_shaders.metal (stale copy backed up).
# Selective taps per DUMP_POLICY: full tensors only for e*/step0_* + b0/b49 at steps 0,1,10,19.
# Arms (GPU serial): EAGER (original bf16 dir) -> NATIVE (model_native_v2, fixed HEAD bdeadcb).
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
PROMPT="一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格"
PROMPT2="連江海岸的燈塔在暴風雨中閃爍，海浪拍打岩岸，寫實攝影風格"
LOG=$REPO/scratch/sol/chain_p4b.log
cd "$REPO" || exit 9

arm() { # TAG MODELDIR [extra envs...]
  TAG=$1; MODEL=$2; shift 2
  D=$EV/$TAG; mkdir -p "$D"
  {
    echo "=== $TAG start $(date) ==="
    echo "--- version stamp ---"
    echo "git_head=$(git -C $REPO rev-parse HEAD)"
    echo "shader_sha256=$(shasum -a 256 "$REPO/h3_shaders.metal" | cut -d' ' -f1)"
    echo "shader_path_compiled=$MODEL/FL2VA/transformer/h3_shaders.metal"
    ls -laL "$MODEL/FL2VA/transformer/h3_shaders.metal" 2>/dev/null
    echo "binary_sha256=$(shasum -a 256 "$REPO/h3" | cut -d' ' -f1)"
    echo "--- args ---"
    echo "model_dir=$MODEL prompt=$PROMPT width=640 height=352 frames=25 steps=20 seed=1 cfg=default"
    echo "envs=$*"
    df -m /Volumes/SSD2T | tail -1
  } > "$D/run_log.txt" 2>&1
  env "$@" \
    H3_DUMP_DIR="$D" H3_DUMP_STEPS=0,1,10,19 H3_DUMP_TAPS="b0_s*,b49_s*,e*,step0_*" \
    /usr/bin/time -l ./h3 -d "$MODEL" \
    -p "$PROMPT" --width 640 --height 352 --frames 25 --steps 20 --seed 1 \
    -o "$D/out.mp4" >> "$D/run_log.txt" 2>&1
  RC=$?
  grep -E "maximum resident|real" "$D/run_log.txt" | sed 's/^/MEMCPU /' >> "$D/run_log.txt"
  echo "ARM_DONE tag=$TAG rc=$RC mp4_bytes=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0)" >> "$D/run_log.txt"
}

{
echo "== CHAIN_P4B start $(date) =="
pgrep -f "./h3 -d" >/dev/null && { echo "GPU BUSY — abort"; exit 8; }
df -m /Volumes/SSD2T | tail -1
echo "== ARM1 EAGER 640 s20 (unified shader, taps) =="
arm p4b_eager640_taps /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3
echo "== ARM2 NATIVE fixed 640 s20 (taps) =="
arm p4b_native640_taps /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 H3_NATIVE_INT8=1
echo "CHAIN_P4B_DONE $(date)"
} >> "$LOG" 2>&1
