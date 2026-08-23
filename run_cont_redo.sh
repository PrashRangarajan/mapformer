#!/usr/bin/env bash
# Re-run of the two continuous-displacement conditions that died of CUDA OOM.
#
# Cause: GPU 1 was carrying the continuous test, the n=8 evaluations and the
# PoPE-Hier job simultaneously. `commanded` (which ran first) survived; the other
# two allocated into an already-full device and every arm hit
# CUBLAS_STATUS_ALLOC_FAILED.
#
# Two changes: wait for the GPU to be free before starting, and run TWO jobs at a
# time instead of six.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/continuous_alloc
# wait until GPU 1 is below 4 GB before starting
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 4000 ]; do sleep 60; done
echo "$(date +%H:%M) gpu1 clear; starting"
CONDS=(
  "allocentric:--n-headings 12 --action-record allocentric"
  "allocnoise:--n-headings 12 --action-record allocentric --heading-noise 0.15"
)
for C in "${CONDS[@]}"; do
  LBL="${C%%:*}"; FLAGS="${C#*:}"
  for SEED in 0 1 2; do
    for V in Vanilla RoPE; do
      D="$R/${LBL}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && { echo "skip $LBL $V s$SEED"; continue; }
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches 980 --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        --grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only \
        $FLAGS --device cuda:1 --output-dir "$D" \
        > "$R/train_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  N=$(find "$R/$LBL" -name '*.pt' 2>/dev/null | wc -l)
  [ "$N" -lt 6 ] && { echo "FAILED $LBL: $N/6"; exit 1; }
  echo "$(date +%H:%M) $LBL done ($N/6)"
done
echo DONE; touch "$R/.done"
