#!/usr/bin/env bash
# CoarsePI: coarse level runs its OWN path integration (disconnected from fine
# cum_delta). Fills the path-angle x disconnected quadrant. Train 3 seeds on
# both tasks, re-aggregate. Disentangles 'spatial coarse pos' from 'pooling noise'.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/coarsepi.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
V=Hourglass_CoarsePI
hg(){ local g=$1 s=$2 o="$REPO/runs/hiergoal_multiseed/seed$2"
  [ -f "$o/${V}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] hg s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$V" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
cp(){ local g=$1 s=$2 o="$REPO/runs/comp_multiseed/seed$2"
  [ -f "$o/${V}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] comp s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$V" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( hg 0 0; hg 0 1; hg 0 2; cp 0 2; echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) & P0=$!
( cp 1 0; cp 1 1;              echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI PlainFlat PlainHourglass \
  --lengths 64 128 192 256 --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants Vanilla VanillaEM Hourglass_k2 Hourglass_CoarseIdx Hourglass_CoarsePI HourglassFlat3 \
             Hourglass_MotifSeg Hourglass_MotifSeg_FR FrameResetFlat PlainHourglass PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.coarsepi_done"
