#!/usr/bin/env bash
# Phase 2 v2 (H3 ingredient 3): train the LOCAL-FRAME-RESET variants on seeds
# {0,1,2}, then re-aggregate ALL variants into COMPOSITIONAL_MULTISEED.md.
#   MapWM-MotifSeg-FR  = MotifSeg + reset (full H3: segmentation + collapse)
#   MapWM-Flat-FR      = flat MapFormer + reset only (isolates the reset)
# Compare: MotifSeg-FR vs MotifSeg (does the reset rescue it?), and
#          Flat-FR vs MotifSeg-FR (reset alone, or reset x hierarchy?).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOG="$REPO/fr.log"; : > "$LOG"; echo "fr start $(date)" >> "$LOG"
OUTROOT="$REPO/runs/comp_multiseed"
VARIANTS=(Hourglass_MotifSeg_FR FrameResetFlat)

JOBS=()
for s in 0 1 2; do for v in "${VARIANTS[@]}"; do JOBS+=("$s:$v"); done; done

train_one () {
  local gpu="$1" job="$2" s="${2%%:*}" v="${2##*:}" out
  out="$OUTROOT/seed$s"
  if [ -f "$out/$v.pt" ]; then echo "$(date +%H:%M) skip $job" >> "$LOG"; return; fi
  echo "$(date +%H:%M) [gpu$gpu] train $job" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif \
    --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" \
    --device "cuda:$gpu" --output-dir "$out" >> "$LOG" 2>&1
}

( for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 0 ] && train_one 0 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) &
P0=$!
( for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 1 ] && train_one 1 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) TRAINING DONE -> aggregating" >> "$LOG"

python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$OUTROOT" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 HourglassFlat3 Hourglass_MotifSeg \
             Hourglass_MotifSeg_FR FrameResetFlat PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) AGGREGATE DONE" >> "$LOG"
touch "$REPO/.fr_done"
