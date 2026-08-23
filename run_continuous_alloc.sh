#!/usr/bin/env bash
# Does allocentric recoding survive CONTINUOUS displacement?
#
# ALLOCENTRIC_RECODING.md: recording the absolute displacement instead of
# turn/forward restores MapFormer completely (+0.049 -> +0.485). Stated limit:
# there the displacement was one of FOUR exact compass symbols. Habitat turns 30
# degrees (TWELVE headings), moves a real-valued 0.25 m, and under the realistic
# setting has actuation noise so the executed rotation differs from the commanded
# one. Whether the recovery survives when the token is an APPROXIMATION of a
# continuous displacement was untested -- and it gates any Habitat port.
#
# Three conditions at H=12, all gated BEFORE training (o1/o3/o5 vs marginal:
# 0.501/0.440/0.462 vs 0.507; 0.462/0.401/0.491 vs 0.497; 0.502/0.512/0.542 vs
# 0.530 -- all PASS):
#   commanded    turn/forward tokens, the known-bad encoding
#   allocentric  displacement quantised into 12 direction bins
#   alloc+noise  same, but the true heading drifts 0.15 rad per turn, so the
#                token is systematically wrong by a growing amount
#
# BUDGET: the scored rate falls to 0.022 (0.016 with noise) against baseline's
# 0.225 -- with 12 headings, straight runs cover more distinct cells and revisit
# less. At the standard budget both arms sat on the floor in the H=4 case
# (KNOB_SWEEP.md), so this uses 980 batches, ~10x, to keep the number of
# gradient-contributing events comparable (rule 5).
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f mapformer/runs/knob_n8/.done ]; do sleep 120; done
echo "$(date +%H:%M) n=8 job finished; starting continuous-displacement test"
R=mapformer/runs/continuous_alloc; mkdir -p "$R"
CONDS=(
  "commanded:--n-headings 12"
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
  echo "$(date +%H:%M) $LBL done"
done
echo DONE; touch "$R/.done"
