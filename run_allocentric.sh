#!/usr/bin/env bash
# THE DECISIVE TEST of why rotation actions collapse the position effect.
#
# KNOB_SWEEP.md: rotate cuts the position effect from +0.478 to +0.049, twice the
# next largest knob. Proposed mechanism: MapFormer path-integrates by cumsumming
# a FIXED per-token delta, and under turn/turn/forward the displacement depends
# on accumulated heading, which that form cannot represent.
#
# This changes ONLY what the token stream RECORDS. Dynamics are identical --
# same trajectories, same observations, byte-identical answer stream (the gates
# come out the same to three decimals). Under --action-record allocentric the
# recorded token is the absolute displacement that actually occurred, or STAY
# when the step produced none. The displacement IS the token, so the cumsum form
# is well-specified again.
#
# PREDICTION, before the run: the position effect recovers from +0.049 toward
# baseline's +0.478. If it does not, the mis-specification account is wrong and
# something else about rotate dynamics is responsible.
#
# Matched budget (392 batches), since the scored rate is 0.054 and the standard
# budget left BOTH arms on the floor.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/knob_sweep
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do
    D="$R/rotate_allocentric/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 16 --n-batches 392 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --grid-size 64 --n-obs-types 16 --action-mode rotate \
      --action-record allocentric --score-moves-only \
      --device cuda:1 --output-dir "$D" \
      > "$R/train_alloc_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done
echo DONE; touch "$R/.alloc_done"
