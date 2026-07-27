#!/usr/bin/env bash
# Decoupled-coarse-position test: MapWM-Hier-CoarseIdx (coarse level uses INDEX
# position, fine spatial angle NOT transmitted up) on BOTH tasks, 3 seeds each,
# then re-aggregate. Prediction: ~= MapWM-Hier on compositional (hierarchy is
# generic there), WORSE on hier-goal OOD (coarse spatial map is load-bearing).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOG="$REPO/coarseidx.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
V=Hourglass_CoarseIdx

hg_one () {  # hierarchical goal task
  local gpu="$1" s="$2" out="$REPO/runs/hiergoal_multiseed/seed$2"
  [ -f "$out/${V}_hiergoal.pt" ] && { echo "$(date +%H:%M) skip hg seed$s" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$gpu] hg seed$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$V" --seed "$s" \
    --epochs 25 --n-batches 64 --T-explore 64 --T-navigate 64 \
    --eval-explore 64 128 192 256 --n-layers 3 --device "cuda:$gpu" \
    --output-dir "$out" >> "$LOG" 2>&1
}
comp_one () {  # compositional task
  local gpu="$1" s="$2" out="$REPO/runs/comp_multiseed/seed$2"
  [ -f "$out/${V}.pt" ] && { echo "$(date +%H:%M) skip comp seed$s" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$gpu] comp seed$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$V" --target motif \
    --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" \
    --device "cuda:$gpu" --output-dir "$out" >> "$LOG" 2>&1
}

( hg_one 0 0; hg_one 0 1; hg_one 0 2; comp_one 0 2; echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) &
P0=$!
( comp_one 1 0; comp_one 1 1;                        echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) TRAINING DONE -> aggregating" >> "$LOG"

python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx HourglassFlat3 Hourglass_MotifSeg \
             Hourglass_MotifSeg_FR FrameResetFlat PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) AGGREGATE DONE" >> "$LOG"
touch "$REPO/.coarseidx_done"