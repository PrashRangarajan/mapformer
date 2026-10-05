#!/usr/bin/env bash
# SCORE_RANK pilot (outside the batch's seeds): (1) reproduction -- Vanilla s0 through train_score_rank, full 900
# epochs, against the stored runs/rank_mi/p0/Vanilla_s0 (bitwise per-epoch losses); (2) timing at 8 concurrent
# (4 per GPU): seeds 100/101 of the four arms, 40 epochs each (same per-epoch work as the 900-epoch recipe);
# (3) timing at 4 concurrent once (2) has finished. Flags = run_rank_mi.sh's.
set -uo pipefail
REPO=/home/prashr/mapformer; LOG="$REPO/runs/score_rank_pilot/pilot.log"; P="$REPO/runs/score_rank_pilot"
mkdir -p "$P"; source "$REPO/lib_driver.sh"; DRV_SPACING=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
go(){ local V=$1 S=$2 E=$3 G=$4 TAG=$5 OUT="$P/$5/${1}_s${2}"; mkdir -p "$OUT"
  OMP_NUM_THREADS=4 drv_launch "$P/$TAG/${V}_s${S}.log" python3 -u -m mapformer.train_score_rank --variant "$V" --seed "$S" \
    --epochs "$E" --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
echo "start $(date)" >> "$LOG"
go Vanilla 0 900 0 repro
go Vanilla 100 40 0 c8; go MapPoPE-Pair 100 40 0 c8; go Vanilla_r4mi 100 40 0 c8
go MapPoPE-Pair_r4mi 100 40 1 c8; go Vanilla 101 40 1 c8; go MapPoPE-Pair 101 40 1 c8; go MapPoPE-Pair_r4mi 101 40 1 c8
drv_wait_dir "$P/c8/"
go Vanilla 100 40 1 c4; go MapPoPE-Pair 100 40 1 c4; go MapPoPE-Pair 101 40 0 c4
drv_wait_dir "$P/c4/"
echo "timing done $(date)" >> "$LOG"
