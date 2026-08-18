#!/usr/bin/env bash
# PoPE on the paper's own task -- completes the 2x2 of
# {encoding scheme} x {path-integrated or index position}.
#
#   RoPE encoding: RoPE (index) 0.516   vs  Vanilla (path int.) 0.989   [done]
#   PoPE encoding: PoPE-Flat (index)    vs  MapPoPE-Flat (path int.)    [this]
#
# Match-Query already shows the same contrast on a gated task (PoPE-Flat 0.117,
# MapPoPE-Hier 0.847), on the CORRECTED PoPE implementation -- the d-band fix
# (3fb40a4) landed 2026-08-09 00:44 and those checkpoints were trained 22:00
# the same day. Running it on the paper's task turns "the axis is path
# integration, not the encoding" from a claim resting on one invented task into
# a factorial on the paper's own benchmark.
#
# Params matched within 0.4%: Vanilla 204,373 / RoPE 203,925 /
# PoPE-Flat 204,053 / MapPoPE-Flat 204,693.
set -euo pipefail
cd "$(dirname "$0")/.."
for SEED in 0 1 2; do
  for VAR in PoPE-Flat MapPoPE-Flat; do
    D="mapformer/runs/paper_task/${VAR}_s${SEED}"
    if [ -f "$D/${VAR}.pt" ]; then echo "skip $VAR s$SEED"; continue; fi
    echo "=== $VAR seed=$SEED ==="
    python3 -u -m mapformer.train_variant --variant "$VAR" --seed "$SEED" \
      --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device cuda:1 --output-dir "$D" \
      > "mapformer/runs/paper_task/train_${VAR}_s${SEED}.log" 2>&1 &
  done
  wait
done
echo TRAINING_DONE
python3 -u -m mapformer.eval_paper_task \
  --variants Vanilla VanillaEM_P0 MapPoPE-Flat RoPE PlainFlat PoPE-Flat \
  --seeds 0 1 2 --device cuda:1 \
  --out mapformer/INDEX_BASELINE_PAPER_TASK.md \
  > mapformer/runs/paper_task/eval_pope.log 2>&1
echo DONE; touch mapformer/runs/paper_task/.pope_done
