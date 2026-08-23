#!/usr/bin/env bash
# The 8th cell of the MiniGrid factorial: PoPE + index + hierarchy, n=8.
#
# WHY IT MATTERS MORE THAN COMPLETENESS: at n=8, PoPE-Flat (index, flat) is the
# BEST arm on this benchmark at T=1024 (0.953 +/- 0.003), and its hierarchy pair
# is exactly the cell that did not exist.
#
# BATCH DISCIPLINE. Rule 3 says do not compare a fresh arm to stored ones. Here
# the usual risk is unusually low -- MiniGrid trains from a FIXED on-disk 25K
# trajectory buffer, so the data is byte-identical, not merely
# distributionally-matched, and no code touching the other seven arms has
# changed. Rather than retrain all 64 runs to prove that, PoPE-Flat is retrained
# alongside as a REPRODUCIBILITY CONTROL: if it reproduces its n=8 number, the
# new cell can be read against the existing grid. This is the same check that
# validated the family-tree and Match-Query batches.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/minigrid_n8
for SEED in 0 1 2 3 4 5 6 7; do
  for V in PoPE-Hier PoPE-Flat-repro; do
    VAR="${V%-repro}"
    D="$R/${V}_s${SEED}"
    [ -f "$D/${VAR}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$VAR" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 \
      --device cuda:1 --output-dir "$D" > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done
echo TRAINED; touch "$R/.pope_hier_done"
