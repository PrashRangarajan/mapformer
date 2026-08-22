#!/usr/bin/env bash
# The torus 2x2 -- {RoPE, PoPE} x {index, path-integrated} -- on a PUBLISHED
# external benchmark, at n=3.
#
# WHY. Everything citable in this repo is the 64x64 torus with one walk
# generator, which is the most predictable reviewer objection there is. MiniGrid
# is external, published, and egocentric (the observation is the cell in front),
# so it is a genuinely different sensory regime.
#
# THE PREDICTION UNDER TEST. MINIGRID_DK16_RESULTS.md, at n=1, has the index
# model BEATING the path-integrating one at long horizon:
#     T=512   Vanilla 0.754   RoPE 0.877
# That inverts the torus result, where RoPE sits on the blank floor (0.514 vs a
# 0.506 floor) and Vanilla scores 0.989.
#
# THE MECHANISM HYPOTHESIS. MiniGrid's actions are 0=turn left, 1=turn right,
# 2=forward -- rotations plus a HEADING-DEPENDENT translation. MapFormer's path
# integrator maps each action token to a fixed Lie-algebra element and cumsums
# it, which assumes actions are translations. On MiniGrid "forward" displaces
# you in a direction set by the accumulated history of turns, so the assumption
# is violated. This is what the paper's own App. B.2.2 motivates with the
# mother/father example -- and then validates on synthetic 4D rotations rather
# than on any real environment.
#
# n=1 cannot support any of that. This is n=3, all four arms in ONE batch
# (rule 3), same recipe as the original run.
#
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/minigrid_2x2
mkdir -p "$R"
VARS="Vanilla MapPoPE-Flat RoPE PoPE-Flat"
GPUQ="1 1 0"; GLOBAL=0

# Build the 25K cached buffer ONCE, serially. Parallel builders would race on
# the same on-disk cache; ~7 min, then ~1.7s/epoch for everything after.
echo "$(date +%H:%M) building cached buffer (one-time, ~7 min)"
python3 -u -m mapformer.train_variant --variant Vanilla --seed 0 \
  --epochs 1 --n-batches 4 --env minigrid_doorkey16 \
  --minigrid-tokenization obj_color --minigrid-cached-buffer 25000 \
  --device cuda:1 --output-dir "$R/_warmup" > "$R/warmup.log" 2>&1
echo "$(date +%H:%M) buffer ready"

for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    G=$(echo $GPUQ | cut -d' ' -f$(( GLOBAL % 3 + 1 ))); GLOBAL=$((GLOBAL + 1))
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done
echo DONE; touch "$R/.done"
