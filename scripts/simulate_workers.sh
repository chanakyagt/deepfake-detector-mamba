#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?usage: simulate_workers.sh <feature_dir> [workers] [out_root]}
N=${2:-4}
OUT=${3:-runs/sim}
CFG=${CFG:-configs/synthetic.yaml}
for spec in "allreduce 1" "diloco 50" "diloco 100" "diloco 500"; do
  set -- $spec
  name="${1}_H${2}"
  torchrun --standalone --nproc_per_node="$N" -m dfmamba.train --config "$CFG" --data "$DATA" \
    --out "$OUT/$name" --mode "$1" --inner-steps "$2" > "$OUT.$name.log" 2>&1
  echo "$name done"
done
