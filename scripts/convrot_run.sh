#!/bin/bash
# convrot_run — build + run the CPU C reference and the Metal engine probe
# for every tensor recorded by scripts/convrot_ref.py, then compare.
# Requires: scripts/convrot_ref.py finished first (dumps q/scale bins + meta.json).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${CONVROT_OUT:-/Users/cyliu99/.hermes/cache/scratch/convrot_sem}"
PY="${H3REF_PY:-/Users/cyliu99/.hermes/cache/scratch/h3ref/bin/python}"
MODEL_DIR="${CONVROT_MODEL_DIR:-/Volumes/SSD2T/models/MiniMax-H3-comfy-int8}"
mkdir -p "$REPO/build" "$OUT"
cd "$REPO"
make -s libh3.a
clang -O3 -o build/convrot_cpu_ref scripts/convrot_cpu_ref.c
clang -O2 -I. -o build/convrot_probe scripts/convrot_probe.c libh3.a \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -framework Accelerate -licucore -lm
"$PY" - "$OUT/meta.json" <<'EOF' > /tmp/convrot_targets.txt
import json, sys
for name, info in json.load(open(sys.argv[1])).items():
    if name.startswith("_"): continue
    n, k = info["shape"]
    print(name, name.replace(".", "_"), n, k, info["groupsize"])
EOF
while read -r name stem rows cols gs; do
  echo "== $stem"
  ./build/convrot_cpu_ref "$OUT" "$stem" "$rows" "$cols" "$gs"
  ./build/convrot_probe "$PWD/h3_shaders.metal" "$MODEL_DIR" "$OUT" "$stem" "$name" "$rows" "$cols"
done < /tmp/convrot_targets.txt
CONVROT_OUT="$OUT" "$PY" scripts/convrot_compare.py
