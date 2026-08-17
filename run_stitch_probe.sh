#!/usr/bin/env bash
# Stitch attention-probe pipeline: train both arms x 3 seeds, then probe.
#
# No `local` anywhere: under `set -u` a `local a=$1 b=$2 c="...$b"` line expands
# every word BEFORE any assignment happens, which has silently mis-parameterised
# launchers in this repo twice.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=mapformer/runs/stitch_probe
mkdir -p "$OUT"

for SEED in 0 1 2; do
  for VAR in MapWM-Flat PlainFlat; do
    if [ "$VAR" = "MapWM-Flat" ]; then GPU=0; else GPU=1; fi
    D="$OUT/${VAR}_s${SEED}"
    if [ -f "$D/${VAR}_stitch.pt" ]; then echo "skip $VAR s$SEED"; continue; fi
    echo "=== train $VAR seed=$SEED on cuda:$GPU ==="
    python3 -m mapformer.train_stitch \
      --variant "$VAR" --seed "$SEED" --epochs 200 --n-batches 48 \
      --device "cuda:$GPU" --output-dir "$D" \
      > "$OUT/train_${VAR}_s${SEED}.log" 2>&1 &
  done
  wait
done

echo "=== probe ==="
python3 -m mapformer.probe_stitch_attention \
  --checkpoints $(ls "$OUT"/*/*_stitch.pt) \
  --n-episodes 200 --device cuda:0 \
  --out mapformer/STITCH_ATTENTION.md > "$OUT/probe.log" 2>&1

echo DONE
touch "$OUT/.done"
