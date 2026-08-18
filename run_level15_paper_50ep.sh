#!/usr/bin/env bash
# Second budget point for the paper-task arm of run_level15_meets_gated.sh.
#
# Why: at the paper's own 16-epoch budget Level15 trains to loss ~0.67 against
# Vanilla's ~0.07. That is very likely undertraining, not a result -- every
# Level 1.5 clean-task claim in CLAUDE.md was measured at 50 EPOCHS, and the
# InEKF's R_t head has to learn a token-type gating function before the
# correction contributes anything. Reading the 16-epoch number as a negative
# would repeat exactly the failure that voided MAP_QUERY_RESULTS.md, where a
# budget copied from another task stopped training before the metric moved
# (standing rule 5).
#
# All three arms get the SAME 50 epochs, so the comparison stays matched at both
# budget points. Waits for the main pipeline so the two do not contend.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/level15_meets
until [ -f "$R/.done" ]; do sleep 60; done
echo "main pipeline finished; starting 50-epoch batch"
for SEED in 0 1 2; do
  for V in Vanilla Vanilla_ExtraHead Level15; do
    D="$R/paper50/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 50 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device cuda:1 --output-dir "$D" \
      > "$R/paper50_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "  50ep seed $SEED done"
done
python3 -u -m mapformer.eval_paper_task --runs-dir "$R/paper50" \
  --variants Vanilla Vanilla_ExtraHead Level15 --seeds 0 1 2 \
  --device cuda:1 --out mapformer/LEVEL15_MEETS_GATED_paper50.md \
  > "$R/eval_paper50.log" 2>&1
echo DONE; touch "$R/.done50"
