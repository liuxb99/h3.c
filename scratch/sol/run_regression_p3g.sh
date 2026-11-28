#!/bin/bash
# Regression gate (p6a): after ANY change to h3_shaders.metal quantize/rotate
# kernels or head-major attention-output path, re-run:
#   1) nohm ground-truth arm + 2) fixed head-major arm at 64x64 subtap
#   3) p3g verdict: b0/b10/b39 out_proj->block_out must be bit-identical (cos=1.00000, relRMS=0)
# and out.mp4 md5 must match between the two arms.
# GPU-serial: pgrep guard is inside run_p3f_pair chain only.
set -u
EV=/Volumes/SSD2T/h3c_evidence_20261126
S10=/Users/cyliu99/.hermes/cache/scratch/s10qa
cd /Users/cyliu99/AI/h3c_mac_20261125/h3c || exit 9
W=64 H=64 bash scratch/sol/run_p3f_pair.sh reg_nohm  "$(pwd)/../model_native_v2" H3_NATIVE_INT8=1 H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1 H3_REFINE_SUBTAP=1
W=64 H=64 bash scratch/sol/run_p3f_pair.sh reg_hm    "$(pwd)/../model_native_v2" H3_NATIVE_INT8=1 H3_REFINE_SUBTAP=1
md5a=$(md5 -q "$EV/reg_nohm/out.mp4"); md5b=$(md5 -q "$EV/reg_hm/out.mp4")
[ "$md5a" = "$md5b" ] || { echo "REGRESSION FAIL: mp4 md5 differ ($md5a vs $md5b)"; exit 1; }
"$S10/../npv/bin/python3" "$S10/p3g_verdict.py" "$EV/reg_hm" | tee "$S10/reg_verdict_out.txt"
if awk '/cos=/{ if ($0 !~ /mlp_adaln/ && $0 !~ /cos= 1\.00000/) { bad=1 } } END{ exit !bad }' "$S10/reg_verdict_out.txt"; then
  echo "REGRESSION FAIL: non-1.0 cos outside mlp_adaln tap $(date)"; exit 1
fi
echo "REGRESSION PASS $(date)"
