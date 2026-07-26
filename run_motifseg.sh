#!/usr/bin/env bash
# Phase 2 (H3): train the ORACLE motif-segmented Hourglass on seeds {0,1,2}
# (same config + output dir as run_comp_multiseed.sh, so it slots into the same
# comparison), then re-aggregate ALL variants into COMPOSITIONAL_MULTISEED.md.
# MotifSeg differs from Hourglass_k2 ONLY in segmentation (rooms vs fixed
# stride), so MotifSeg vs MapWM-Hier isolates H3's segmentation ingredient.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOG="$REPO/motifseg.log"; : > "$LOG"
echo "motifseg start $(date)" >> "$LOG"
OUTROOT="$REPO/runs/comp_multiseed"
V=Hourglass_MotifSeg

train_one () {
  local gpu="$1" s="$2" out="$OUTROOT/seed$2"
  if [ -f "$out/$V.pt" ]; then echo "$(date +%H:%M) skip seed$s (exists)" >> "$LOG"; return; fi
  echo "$(date +%H:%M) [gpu$gpu] train $V seed$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$V" --target motif \
    --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 \
    --seed "$s" --device "cuda:$gpu" --output-dir "$out" >> "$LOG" 2>&1
}

( train_one 0 0; train_one 0 2; echo "$(date +%H:%M) GPU0 chain DONE" >> "$LOG" ) &
P0=$!
( train_one 1 1;                 echo "$(date +%H:%M) GPU1 chain DONE" >> "$LOG" ) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) TRAINING DONE -> aggregating all variants" >> "$LOG"

python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$OUTROOT" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 HourglassFlat3 Hourglass_MotifSeg \
             PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) AGGREGATE DONE" >> "$LOG"
touch "$REPO/.motifseg_done"
