#!/usr/bin/env bash
# LEARNABILITY DIAGNOSTIC: is MiniWorld learnable once we stop undersizing/
# undertraining? Reuses the ALREADY-BUILT G=8 seed-0 buffers (--buffer-size 3000
# hits the cache; no rebuild). Scales d=128->256, 3->4 layers, 2->4 heads (~5M,
# IRIS's lower band) and epochs 40->150 (the ep-40 loss was still dropping).
# Two arms: Vanilla (path-int) vs RoPE (index), raw. If Vanilla climbs well above
# its 0.183 nb_acc AND separates from RoPE, the task is learnable -> scale the
# full factorial to this config. Runs one arm per GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_scale_diag"; mkdir -p "$R"
LOG="$REPO/mw_scale_diag.log"; echo "scale-diag start $(date)" > "$LOG"
COMMON="--grid-size 8 --n-steps 512 --buffer-size 3000 --seed 0 \
  --epochs 150 --n-batches 180 --batch-size 24 \
  --d-model 256 --n-layers 4 --n-heads 4 --eval-lengths 512 1024"

python3 -u -m mapformer.train_miniworld --variant Vanilla $COMMON \
  --device cuda:0 --output-dir "$R" > "$R/Vanilla_raw.log" 2>&1 &
python3 -u -m mapformer.train_miniworld --variant RoPE $COMMON \
  --device cuda:1 --output-dir "$R" > "$R/RoPE_raw.log" 2>&1 &
wait
echo "$(date) diag training done" >> "$LOG"
for V in Vanilla RoPE; do
  python3 -c "
import json; r=json.load(open('$R/${V}_raw.json'))
print('${V}: nb_acc T=512', round(r['512']['nb_acc'],3), 'nb_nll', round(r['512']['nb_nll'],3),
      '| T=1024', round(r['1024']['nb_acc'],3))" >> "$LOG" 2>&1
done
touch "$REPO/.mw_scale_diag_done"
echo "$(date) DONE" >> "$LOG"
cat "$LOG" | tail -6
