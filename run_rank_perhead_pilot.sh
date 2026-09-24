#!/usr/bin/env bash
# Per-head r=2 pilot. Pre-registration: RANK_PERHEAD_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_perhead_pilot.log"
exec 9>"$REPO/.run_rank_perhead.lock"
flock -n 9 || { echo "$(date) REFUSED: lock held" >> "$LOG"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_perhead_pilot"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
( cd "$REPO" && md5sum model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py ) > "$R/code_md5.txt"
fail(){ echo "$(date) FAILED: $1 -- done marker NOT set" >> "$LOG"; exit 1; }
launch(){ local V=$1 S=$2 G=$3; OUT="$R/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state > "$R/${V}_s${S}.log" 2>&1 &
  sleep 20; }
launch Vanilla_r2ph 0 0; launch Vanilla_r2ph 1 1; launch Vanilla 0 1
while [ "$(ps -u "$USER" -o comm=,args= | awk -v r="$R/" '$1=="python3" && /mapformer\.train_variant/ && index($0,r)' | wc -l)" -gt 0 ]; do sleep 60; done
for x in Vanilla_r2ph_s0/Vanilla_r2ph Vanilla_r2ph_s1/Vanilla_r2ph Vanilla_s0/Vanilla; do [ -f "$R/p0/$x.pt" ] || fail "missing $x"; done
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla_r2ph --noises 0.0 --seeds 0 1 \
  --lengths 512 1024 2048 --n-trials 100 --device cuda:0 --title "Per-head r=2 pilot, trained and tested at T=1024" \
  --out "$REPO/RANK_PERHEAD_PILOT.md" >> "$LOG" 2>&1 || fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla_r2ph --seeds 0 1 --lengths 1024 \
  --n-trials 100 --device cuda:0 --check-json "$REPO/RANK_PERHEAD_PILOT.json" --out "$REPO/RANK_PERHEAD_PILOT_STRATA.json" >> "$LOG" 2>&1 || fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla_r2ph --seeds 0 1 --device cuda:0 \
  --out "$REPO/RANK_PERHEAD_PILOT_GEOMETRY.md" >> "$LOG" 2>&1 || fail geometry
python3 -u -m mapformer.analyze_rank_perhead > "$REPO/RANK_PERHEAD_PILOT_ANALYSIS.txt" 2>&1 || fail analyze
touch "$REPO/.rank_perhead_pilot_done"; echo "$(date) DONE" >> "$LOG"
