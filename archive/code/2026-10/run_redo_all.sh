#!/usr/bin/env bash
# Re-run of everything that died: the two continuous-displacement conditions
# (out-of-range token bug) and PoPE-Hier seeds 2-3 (genuine CUDA OOM during the
# window when three jobs shared GPU 1).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== PoPE-Hier seeds 2,3 ==="
R1=mapformer/runs/minigrid_n8
for SEED in 2 3; do
  for V in PoPE-Hier PoPE-Flat-repro; do
    VAR="${V%-repro}"; D="$R1/${V}_s${SEED}"
    [ -f "$D/${VAR}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$VAR" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-cached-buffer 25000 --device cuda:1 --output-dir "$D" \
      > "$R1/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
done
N=$(find "$R1" -path '*PoPE-Hier_s*' -name 'PoPE-Hier.pt' | wc -l)
[ "$N" -lt 8 ] && { echo "FAILED PoPE-Hier: $N/8"; exit 1; }
echo "PoPE-Hier complete ($N/8)"

echo "=== continuous displacement, 2 conditions ==="
R2=mapformer/runs/continuous_alloc
for C in "allocentric:--n-headings 12 --action-record allocentric" \
         "allocnoise:--n-headings 12 --action-record allocentric --heading-noise 0.15"; do
  LBL="${C%%:*}"; FLAGS="${C#*:}"
  for SEED in 0 1 2; do
    for V in Vanilla RoPE; do
      D="$R2/${LBL}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && continue
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches 980 --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        --grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only \
        $FLAGS --device cuda:1 --output-dir "$D" \
        > "$R2/train_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  N=$(find "$R2/$LBL" -name '*.pt' 2>/dev/null | wc -l)
  [ "$N" -lt 6 ] && { echo "FAILED $LBL: $N/6"; exit 1; }
  echo "$LBL complete ($N/6)"
done
echo DONE; touch "$R2/.done"; touch mapformer/runs/.redo_done
