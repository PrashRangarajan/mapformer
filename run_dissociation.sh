#!/usr/bin/env bash
# The dissociation sweep. See sweep_dissociation.py for the pre-registered
# predictions -- they are written down BEFORE the run, in the file, on purpose.
#
# Grid: 4 variants x 4 n_templates x 3 seeds = 48 runs, ~27 min each.
# Six at a time across both GPUs -> ~4 h. GPU 0 is used because both other
# users' jobs are currently near-idle (1% util, 4 GB); GPU 1 carries the same
# load so neither host job is crowded out.
#
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/dissociation
mkdir -p "$R"
VARS="Vanilla Hourglass_k2 PlainFlat PlainHourglass"

for NT in 2 4 8 16; do
  for SEED in 0 1 2; do
    I=0
    for V in $VARS; do
      D="$R/nt${NT}/seed${SEED}"
      if [ -f "$D/${V}.pt" ]; then echo "skip nt$NT $V s$SEED"; continue; fi
      if [ $((I % 2)) -eq 0 ]; then G=0; else G=1; fi
      I=$((I + 1))
      python3 -u -m mapformer.train_compositional \
        --variant "$V" --target motif --n-steps 256 \
        --epochs 50 --n-batches 156 --n-layers 3 \
        --n-templates "$NT" --seed "$SEED" \
        --device "cuda:$G" --output-dir "$D" \
        > "$R/train_nt${NT}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
    echo "$(date +%H:%M) nt=$NT seed=$SEED done"
  done
done

echo "=== evaluating ==="
python3 -u -m mapformer.sweep_dissociation \
  --runs-dir "$R" --templates 2 4 8 16 --seeds 0 1 2 \
  --lengths 256 1024 --device cuda:1 \
  --out mapformer/DISSOCIATION_SWEEP.md > "$R/eval.log" 2>&1
echo DONE; touch "$R/.done"
