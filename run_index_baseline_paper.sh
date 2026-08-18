#!/usr/bin/env bash
# Index-position baselines on the PAPER'S OWN task.
#
# The ablation (PAPER_TASK_ABLATION.md) showed the trained models USE the action
# stream, but an ablation cannot say whether a model COULD solve the task without
# path integration. That needs a model that never had it. Same recipe, same
# hyperparameters, same seeds as the Vanilla / VanillaEM_P0 runs already in
# runs/paper_task -- only the position code differs.
#
#   RoPE       MapFormer-WM with theta = t * freqs instead of omega*cumsum(Delta).
#              The tightest control: identical architecture, position code swapped.
#   PlainFlat  ordinary transformer, index RoPE. The control used in Match-Query,
#              included so the two tasks are compared through the same baseline.
#
# Params are matched within 0.2%: Vanilla 204,373 / RoPE 203,925 / PlainFlat 203,925.
# No `local` anywhere -- under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
for SEED in 0 1 2; do
  for VAR in RoPE PlainFlat; do
    D="mapformer/runs/paper_task/${VAR}_s${SEED}"
    if [ -f "$D/${VAR}.pt" ]; then echo "skip $VAR s$SEED"; continue; fi
    echo "=== $VAR seed=$SEED ==="
    python3 -u -m mapformer.train_variant --variant "$VAR" --seed "$SEED" \
      --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device cuda:1 --output-dir "$D" \
      > "mapformer/runs/paper_task/train_${VAR}_s${SEED}.log" 2>&1 &
  done
  wait
done
echo TRAINING_DONE
python3 -u -m mapformer.eval_paper_task \
  --variants Vanilla VanillaEM_P0 RoPE PlainFlat --seeds 0 1 2 \
  --device cuda:1 --out mapformer/INDEX_BASELINE_PAPER_TASK.md \
  > mapformer/runs/paper_task/eval_index.log 2>&1
echo DONE; touch mapformer/runs/paper_task/.index_done
