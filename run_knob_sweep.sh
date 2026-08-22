#!/usr/bin/env bash
# WHICH environment property flips the sign of the position effect?
#
# Established: path integration is worth +0.461 on the torus paper task
# (INDEX_BASELINE_PAPER_TASK_n8.md) and -0.060 on MiniGrid-DoorKey-16x16
# (FREQ_CONTROL.md, with the frequency confound ruled out). The two environments
# differ in FIVE ways at once, so which one is responsible is unknown. This turns
# them ONE AT A TIME from the torus baseline, plus the full combination.
#
# The measured quantity is the POSITION EFFECT: Vanilla (path-integrated) minus
# RoPE (index), both trained in the same batch at every condition (rule 3).
#
# PRE-REGISTERED PREDICTION, written before the run: aliasing and size drive it,
# the embodiment knobs (rotate, ego, wall) do not. Reasoning: the torus index
# arms fail by sitting on a blank floor that only exists because observations are
# ambiguous -- with 16 obs types over 4096 cells there is nothing to localise
# from, so position is the only route. Shrink the map or enrich the observations
# and the content route opens.
#
# The interesting alternative: if ROTATE is what kills it, then MapFormer's
# SO(2) cumsum is mis-specified for the action space most embodied agents have,
# which is a sharper and more consequential finding.
#
# GPU 1 only. No `local` anywhere (word-expansion under set -u).
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/knob_sweep
mkdir -p "$R"

# label : extra flags
CONDS=(
  "baseline:--grid-size 64 --n-obs-types 16"
  "rotate:--grid-size 64 --n-obs-types 16 --action-mode rotate"
  "ego:--grid-size 64 --n-obs-types 16 --obs-mode ego"
  "wall:--grid-size 64 --n-obs-types 16 --boundary wall"
  "small:--grid-size 16 --n-obs-types 16"
  "richobs:--grid-size 64 --n-obs-types 64"
  "allcombined:--grid-size 16 --n-obs-types 64 --action-mode rotate --obs-mode ego --boundary wall"
)

for C in "${CONDS[@]}"; do
  LBL="${C%%:*}"; FLAGS="${C#*:}"
  for SEED in 0 1 2; do
    for V in Vanilla RoPE; do
      D="$R/${LBL}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && { echo "skip $LBL $V s$SEED"; continue; }
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        $FLAGS --device cuda:1 --output-dir "$D" \
        > "$R/train_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  echo "$(date +%H:%M) $LBL done"
done
echo TRAINING_DONE; touch "$R/.trained"
