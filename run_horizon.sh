#!/usr/bin/env bash
# Horizon measurement, part 1: is the ~2-step attention horizon ARCHITECTURAL or
# just capacity?
#
# REVISIT_DISTANCE.md measured that index-position models beat the blank floor
# only at recurrence interval 1-2 and nowhere else. Part 2
# (HORIZON_TASK_DISTANCES.md) tested whether that one number predicts cross-task
# performance and it DOES NOT -- so the remaining well-posed question is the
# within-task one, where floors and aliasing are held fixed:
#
#   does h grow with depth, width, or training budget?
#     saturates -> architectural bound (the strong result)
#     grows     -> a capacity story (weaker, still publishable, honest either way)
#
# Grid: RoPE (index, architecture-matched to MapFormer-WM) and Vanilla (the
# path-integrated reference, whose curve should stay flat) across 5 configs x 3
# seeds, plus a 50-epoch budget point at the base config.
#
# Waits for the seed scale-up so nothing contends.
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/horizon
until [ -f mapformer/runs/paper_task_n8/.done ]; do sleep 120; done
echo "$(date +%H:%M) seed scale-up finished; starting horizon grid"
mkdir -p "$R"

# label:layers:dmodel:epochs
CONFIGS="L1d128e16:1:128:16 L2d128e16:2:128:16 L4d128e16:4:128:16 L2d256e16:2:256:16 L4d256e16:4:256:16 L1d128e50:1:128:50"

for C in $CONFIGS; do
  LBL=$(echo "$C" | cut -d: -f1); NL=$(echo "$C" | cut -d: -f2)
  DM=$(echo "$C" | cut -d: -f3);  EP=$(echo "$C" | cut -d: -f4)
  for SEED in 0 1 2; do
    I=0
    for V in RoPE Vanilla; do
      D="$R/$LBL/${V}_s${SEED}"
      if [ -f "$D/${V}.pt" ]; then echo "skip $LBL $V s$SEED"; continue; fi
      if [ $((I % 2)) -eq 0 ]; then G=0; else G=1; fi
      I=$((I + 1))
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs "$EP" --n-batches 98 --batch-size 128 --n-steps 128 \
        --n-layers "$NL" --n-heads 2 --d-model "$DM" --n-landmarks 0 \
        --device "cuda:$G" --output-dir "$D" \
        > "$R/train_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  echo "$(date +%H:%M) $LBL done"
done

echo "=== measuring the horizon per config ==="
for C in $CONFIGS; do
  LBL=$(echo "$C" | cut -d: -f1)
  python3 -u -m mapformer.probe_revisit_distance \
    --runs-dir "$R/$LBL" --variants RoPE Vanilla --seeds 0 1 2 \
    --device cuda:1 --out "mapformer/HORIZON_${LBL}.md" \
    > "$R/probe_${LBL}.log" 2>&1
done
echo DONE; touch "$R/.done"
