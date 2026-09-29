#!/usr/bin/env bash
# H1 at 1800 epochs, part 1 (A and C only). Pre-registration: LOOP_RANK_E1800_PREREG.md, Amendment 1.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/loop_rank_e1800.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
DRV_MODULE_RE="mapformer[.]train_(variant|ctxstep)"
drv_lock "$REPO/.run_loop_rank_e1800.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/loop_rank_e1800"; mkdir -p "$R/p0"
echo "start part 1 (A, C) $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_looped.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py || exit 1
launch(){ local V=$1 S=$2 G; OUT="$R/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 1800 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
for S in 0 1 2 3 4 5 6 7; do launch Vanilla $S; launch Vanilla_r4mi $S; done
drv_wait_dir "$R/"
REQ=(); for S in 0 1 2 3 4 5 6 7; do for V in Vanilla Vanilla_r4mi; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla Vanilla_r4mi --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "H1 part 1: rank 2 and rank 4 at 1800 epochs, T=1024" --out "$REPO/LOOP_RANK_E1800_P1.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.analyze_loop_rank_e1800_budget > "$REPO/LOOP_RANK_E1800_P1_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.loop_rank_e1800_p1_done" "$REPO/LOOP_RANK_E1800_P1.md" "$REPO/LOOP_RANK_E1800_P1.json" "$REPO/LOOP_RANK_E1800_P1_ANALYSIS.txt"
