#!/usr/bin/env bash
# Rank at matched initialisation. Pre-registration: RANK_MI_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_mi.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
drv_lock "$REPO/.run_rank_mi.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_mi"; RP="$REPO/runs/rank_mi_repro"; mkdir -p "$R/p0" "$RP/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py || exit 1
# reuse (RANK_MI_PREREG.md): stored A s0-7 and the pilot's B s0-1, linked, never copied or moved
for S in 0 1 2 3 4 5 6 7; do [ -e "$R/p0/Vanilla_s$S" ] || ln -s "$REPO/runs/rank_matched_e900/p0/Vanilla_s$S" "$R/p0/Vanilla_s$S"; done
for S in 0 1; do [ -e "$R/p0/Vanilla_r2ph_s$S" ] || ln -s "$REPO/runs/rank_perhead_pilot/p0/Vanilla_r2ph_s$S" "$R/p0/Vanilla_r2ph_s$S"; done
launch(){ local ROOT=$1 V=$2 S=$3 G; OUT="$ROOT/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
launch "$RP" Vanilla 3
for S in 0 1 2 3 4 5 6 7; do launch "$R" Vanilla_r4mi $S; [ "$S" -ge 2 ] && launch "$R" Vanilla_r2ph $S; done
drv_wait_dir "$R/"; drv_wait_dir "$RP/"
REQ=("$RP/p0/Vanilla_s3/Vanilla.pt")
for S in 0 1 2 3 4 5 6 7; do for V in Vanilla Vanilla_r2ph Vanilla_r4mi; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla Vanilla_r2ph Vanilla_r4mi --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank at matched initialisation, trained and tested at T=1024" --out "$REPO/RANK_MI.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla Vanilla_r2ph Vanilla_r4mi --seeds 0 1 2 3 4 5 6 7 \
  --lengths 1024 --n-trials 100 --device cuda:0 --check-json "$REPO/RANK_MI.json" --out "$REPO/RANK_MI_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla Vanilla_r2ph Vanilla_r4mi --seeds 0 1 2 3 4 5 6 7 \
  --space delta --device cuda:0 --out "$REPO/RANK_MI_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail geometry
python3 -u -m mapformer.analyze_rank_mi > "$REPO/RANK_MI_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_mi_done" "$REPO/RANK_MI.md" "$REPO/RANK_MI.json" "$REPO/RANK_MI_STRATA.json" \
  "$REPO/RANK_MI_GEOMETRY.md" "$REPO/RANK_MI_ANALYSIS.txt"
