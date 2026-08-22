#!/usr/bin/env bash
# n=3 -> n=8 on the cells that carry the new headline.
#
# WHY THIS AND NOT MORE RESULTS. Three of the four blockers named this morning
# are closed (second environment family, the dissociation characterised, lm200
# gated). Statistical power is the one that is left, and it is the one that has
# actually bitten: three seeds misled us four times in a single day -- the
# compositional outlier that inverted a conclusion, the family-tree seed that
# manufactured a +0.038, the unstable `small` and `richobs` arms, and MapPoPE's
# "non-overlapping ranges" evaporating when n went 3 -> 8.
#
# ALL EIGHT SEEDS ARE TRAINED FRESH, including 0-2 which already exist. Pooling
# five new seeds into three stored ones is the fresh-vs-stored comparison rule 3
# forbids; these runs are cheap enough that there is no reason to take the risk.
# New directories, nothing shared with the n=3 runs.
#
# Scope is the load-bearing subset, not everything: the MiniGrid factorial (which
# carries "the ingredient ranking reorders") and the four knob conditions that
# carry "rotation actions are why, and here is the fix". The remaining knob
# conditions stay at n=3 with the count stated.
#
# GPU 1 only -- another user is on GPU 0.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="0 1 2 3 4 5 6 7"

echo "=== PHASE 1: MiniGrid factorial, 7 variants x 8 seeds ==="
R1=mapformer/runs/minigrid_n8; mkdir -p "$R1"
for SEED in $SEEDS; do
  for V in Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat; do
    D="$R1/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 \
      --device cuda:1 --output-dir "$D" > "$R1/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) minigrid seed $SEED done"
done
touch "$R1/.done"

echo "=== PHASE 2: knob conditions, 4 x 2 variants x 8 seeds ==="
R2=mapformer/runs/knob_n8; mkdir -p "$R2"
CONDS=(
  "baseline:98:--grid-size 64 --n-obs-types 16"
  "allcombined:98:--grid-size 16 --n-obs-types 64 --action-mode rotate --obs-mode ego --boundary wall"
  "rotate:392:--grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only"
  "allocentric:392:--grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only --action-record allocentric"
)
for C in "${CONDS[@]}"; do
  LBL="${C%%:*}"; REST="${C#*:}"; NB="${REST%%:*}"; FLAGS="${REST#*:}"
  for SEED in $SEEDS; do
    for V in Vanilla RoPE; do
      D="$R2/${LBL}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && { echo "skip $LBL $V s$SEED"; continue; }
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches "$NB" --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        $FLAGS --device cuda:1 --output-dir "$D" \
        > "$R2/train_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  echo "$(date +%H:%M) knob $LBL done"
done
echo DONE; touch "$R2/.done"
