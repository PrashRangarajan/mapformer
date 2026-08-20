#!/usr/bin/env bash
# n=3 -> n=8 on the paper-task arms, so the headline OOD table stops resting on
# three seeds (standing rule 6: three seeds is not a point estimate).
#
# ALL EIGHT SEEDS ARE TRAINED FRESH, including 0-2 which already exist. Mixing
# five new seeds into three stored ones would be exactly the fresh-vs-stored
# comparison rule 3 forbids, and the stored ones cost 2 minutes each to redo, so
# there is no reason to take the risk. New directory, one batch, nothing shared
# with runs/paper_task.
#
# Six variants = the full 2x2 of {RoPE, PoPE} x {index, path-integrated} plus the
# two EM arms the OOD table compares against.
#
# Waits for the dissociation sweep so the two do not contend for GPUs.
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/paper_task_n8
until [ -f mapformer/runs/dissociation/.done ]; do sleep 120; done
echo "$(date +%H:%M) dissociation sweep finished; starting seed scale-up"
mkdir -p "$R"
VARS="Vanilla VanillaEM_P0 MapPoPE-Flat RoPE PlainFlat PoPE-Flat"

for SEED in 0 1 2 3 4 5 6 7; do
  I=0
  for V in $VARS; do
    D="$R/${V}_s${SEED}"
    if [ -f "$D/${V}.pt" ]; then echo "skip $V s$SEED"; continue; fi
    if [ $((I % 2)) -eq 0 ]; then G=0; else G=1; fi
    I=$((I + 1))
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done"
done

echo "=== eval: extended OOD (the headline table) ==="
python3 -u -m mapformer.eval_paper_ood --runs-dir "$R" \
  --variants Vanilla VanillaEM_P0 MapPoPE-Flat --seeds 0 1 2 3 4 5 6 7 \
  --extended --n-batches 8 --batch-size 32 --device cuda:1 \
  --out mapformer/PAPER_OOD_EXTENDED_n8.md > "$R/eval_ood.log" 2>&1

echo "=== eval: the 2x2 ==="
python3 -u -m mapformer.eval_paper_task --runs-dir "$R" \
  --variants Vanilla MapPoPE-Flat RoPE PlainFlat PoPE-Flat VanillaEM_P0 \
  --seeds 0 1 2 3 4 5 6 7 --device cuda:1 \
  --out mapformer/INDEX_BASELINE_PAPER_TASK_n8.md > "$R/eval_2x2.log" 2>&1
echo DONE; touch "$R/.done"
