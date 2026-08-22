#!/usr/bin/env bash
# Frequency control on MiniGrid, where the position effect is small enough for
# this confound to account for it (-0.012 to -0.037).
#
# Vanilla vs Vanilla_FixedOmega isolates FREQUENCY LEARNING exactly: identical
# architecture and init, path integration intact, only the 64 omega values
# frozen. Vanilla and RoPE are retrained here too so all three are one batch
# (rule 3) rather than the new arm being read against the grid from an hour ago.
#
# GPU 1 only -- default for this session; GPU 0 is left to other users.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/freq_control
mkdir -p "$R"
for SEED in 0 1 2; do
  for V in Vanilla Vanilla_FixedOmega RoPE; do
    D="$R/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 \
      --device cuda:1 --output-dir "$D" \
      > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done
python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir "$R" \
  --variants Vanilla Vanilla_FixedOmega RoPE --seeds 0 1 2 \
  --lengths 128 512 1024 --device cuda:1 \
  --out mapformer/FREQ_CONTROL.md > "$R/eval.log" 2>&1
echo DONE; touch "$R/.done"
