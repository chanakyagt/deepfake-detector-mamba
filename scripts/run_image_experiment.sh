#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?usage: run_image_experiment.sh <feature_dir> [out_dir]}
OUT=${2:-runs/images}
mkdir -p "$OUT"
python -m dfmamba.train --config configs/images.yaml --data "$DATA" --out "$OUT/single_H1" --mode single > "$OUT/single_H1.log" 2>&1
python -m dfmamba.train --config configs/images_no_temporal.yaml --data "$DATA" --out "$OUT/single_no_temporal" --mode single > "$OUT/single_no_temporal.log" 2>&1
for spec in "allreduce 1" "diloco 100" "diloco 500"; do
  set -- $spec
  torchrun --standalone --nproc_per_node=4 -m dfmamba.train --config configs/images.yaml --data "$DATA" \
    --out "$OUT/${1}_H${2}" --mode "$1" --inner-steps "$2" > "$OUT/${1}_H${2}.log" 2>&1
done
python scripts/summarize_runs.py "$OUT"
