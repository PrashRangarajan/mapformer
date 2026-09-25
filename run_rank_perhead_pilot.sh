#!/usr/bin/env bash
# Per-head r=2 pilot. Pre-registration: RANK_PERHEAD_PREREG.md.
# Scheduling, locking, the md5 guard and the done marker come from lib_driver.sh (2026-09-24).
# The md5 is now a GUARD (was a record): a re-run with changed training code ABORTS.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_perhead_pilot.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
drv_lock "$REPO/.run_rank_perhead.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_perhead_pilot"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py || exit 1
launch(){ local V=$1 S=$2 G; OUT="$R/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
launch Vanilla_r2ph 0; launch Vanilla_r2ph 1; launch Vanilla 0
drv_wait_dir "$R/"
drv_require "$R/p0/Vanilla_r2ph_s0/Vanilla_r2ph.pt" "$R/p0/Vanilla_r2ph_s1/Vanilla_r2ph.pt" "$R/p0/Vanilla_s0/Vanilla.pt" \
  || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla_r2ph --noises 0.0 --seeds 0 1 \
  --lengths 512 1024 2048 --n-trials 100 --device cuda:0 --title "Per-head r=2 pilot, trained and tested at T=1024" \
  --out "$REPO/RANK_PERHEAD_PILOT.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla_r2ph --seeds 0 1 --lengths 1024 \
  --n-trials 100 --device cuda:0 --check-json "$REPO/RANK_PERHEAD_PILOT.json" --out "$REPO/RANK_PERHEAD_PILOT_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla_r2ph --seeds 0 1 --device cuda:0 \
  --out "$REPO/RANK_PERHEAD_PILOT_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail geometry
python3 -u -m mapformer.analyze_rank_perhead > "$REPO/RANK_PERHEAD_PILOT_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_perhead_pilot_done" "$REPO/RANK_PERHEAD_PILOT.md" "$REPO/RANK_PERHEAD_PILOT.json" \
  "$REPO/RANK_PERHEAD_PILOT_STRATA.json" "$REPO/RANK_PERHEAD_PILOT_GEOMETRY.md" "$REPO/RANK_PERHEAD_PILOT_ANALYSIS.txt"
