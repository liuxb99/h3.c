#!/bin/bash
# P4F: post-patch native acceptance run (same contract as chain_p4b native arm: 640x352x25f steps20 seed1 prompt1).
# Acceptance: refiner taps must now be BIT-IDENTICAL to p4b_eager640_taps (golden) refs, since refiner/passthrough
# weights are now golden bytes and the code path is shared bf16 between arms.
# Waits for CPU byte-verify (verify_postpatch.out containing VERIFY:) + PASS line before touching GPU.
set -u
REPO=/Users/cyliu99/AI/h3c_mac_20261125/h3c
EV=/Volumes/SSD2T/h3c_evidence_20261126
S10=/Users/cyliu99/.hermes/cache/scratch/s10qa
PROMPT="一只橘色小貓在草地上歡快地追逐蝴蝶，陽光明媚，卡通風格"
LOG=$REPO/scratch/sol/chain_p4f.log
cd "$REPO" || exit 9
{
echo "=== P4F start $(date) ==="
# 1) wait for byte verify result (pid 19951) up to 40 min
for i in $(seq 1 240); do
  [ -f "$S10/verify_postpatch.out" ] && grep -q "VERIFY:" "$S10/verify_postpatch.out" && break
  sleep 10
done
grep -q "bad=0" "$S10/verify_postpatch.out" || { echo "BYTE_VERIFY_FAIL — abort GPU run"; cat "$S10/verify_postpatch.out"; exit 7; }
echo "BYTE_VERIFY_OK $(grep VERIFY: "$S10/verify_postpatch.out")"
# 2) GPU guards: wait for ComfyX queue idle up to 4 h (poll 30 s); ComfyUI /queue JSON busy iff queue_running/pending non-empty
idle () {
  local q
  q=$(curl -s -m3 http://127.0.0.1:8390/queue 2>/dev/null) || return 0
  [ -z "$q" ] && return 0
  echo "$q" | python3 -c 'import json,sys
d=json.load(sys.stdin)
sys.exit(0 if not d.get("queue_running") and not d.get("queue_pending") else 1)' 2>/dev/null
}
for i in $(seq 1 480); do
  pgrep -f "./h3 -d" && { echo GPU_BUSY; exit 8; }
  idle && break
  sleep 30
done
idle || { echo GPU_BUSY_COMFYX_TIMEOUT; exit 8; }
D=$EV/p4f_nativepost640; mkdir -p "$D"
echo "git_head=$(git -C $REPO rev-parse HEAD)"
echo "binary_sha256=$(shasum -a 256 "$REPO/h3" | cut -d' ' -f1)"
df -m /Volumes/SSD2T | tail -1
env H3_DUMP_DIR="$D" H3_DUMP_STEPS=0,1,10,19 H3_DUMP_TAPS="b0_s*,b49_s*,e*,step0_*" \
  /usr/bin/time -l ./h3 -d /Users/cyliu99/AI/h3c_mac_20261125/model_native_v2 \
  -p "$PROMPT" --width 640 --height 352 --frames 25 --steps 20 --seed 1 \
  -o "$D/out.mp4" > "$D/run_log.txt" 2>&1
RC=$?
grep -E "maximum resident|real" "$D/run_log.txt" | sed 's/^/MEMCPU /' >> "$D/run_log.txt"
echo "ARM_DONE tag=p4f_nativepost640 rc=$RC raw=$(ls "$D"/*.raw 2>/dev/null | wc -l) mp4_bytes=$([ -s "$D/out.mp4" ] && stat -f%z "$D/out.mp4" || echo 0)"
echo "CHAIN_P4F_DONE $(date)"
} >> "$LOG" 2>&1
