#!/usr/bin/env bash
# The three arms FAMILY_TREE_RESULTS.md already has, retrained in the SAME batch
# as the new Vanilla and Level15 arms (rule 3). Without this the striking
# comparison -- plain WM at ~0.84 against that file's best entry of 0.729 -- is
# cross-batch and worth nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/correction_gaps
for SEED in 0 1 2; do
  I=0
  for V in MapEM_NC_NL VanillaEM_P0 PlainFlat; do
    D="$R/familytree/seed${SEED}"
    [ -f "$D/${V}_familytree.pt" ] && { echo "skip $V s$SEED"; continue; }
    if [ $((I % 2)) -eq 0 ]; then G=0; else G=1; fi
    I=$((I + 1))
    python3 -u -m mapformer.train_family_tree --variant "$V" --seed "$SEED" \
      --epochs 100 --n-batches 48 --batch-size 16 --depth 5 --n-obs 8 \
      --n-steps 64 --eval-steps 64 128 --n-layers 2 \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/ft_${V}_s${SEED}.log" 2>&1 &
  done
  wait
done
echo FT_EXTRA_DONE; touch "$R/.ft_extra_done"
