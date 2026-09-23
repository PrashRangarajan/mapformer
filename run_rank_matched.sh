#!/usr/bin/env bash
# Rank at matched length. Pre-registration: RANK_MATCHED_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
exec 9>"$REPO/.run_rank_matched.lock"
flock -n 9 || { echo "another instance of $(basename "$0") is already running -- exiting"; exit 0; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_matched"; mkdir -p "$R/p0"
LOG="$REPO/rank_matched.log"; echo "start $(date)" >> "$LOG"
ARMS="Vanilla Vanilla_r4"; SEEDS="0 1 2 3 4 5 6 7"
MAXPG=5
# count REAL trainers on a device (ps comm=python3; never pgrep -f, which matches shells)
ntrain(){ ps -u "$USER" -o comm=,args= | awk -v d="--device cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }
freemem(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
# pick the less loaded device that has a slot AND ~4 GiB free (explicit attention at
# 16 x 2 x 2047^2 fp32 is ~3 GB per job)
pick(){ local best="" bn=99; for g in 0 1; do n=$(ntrain $g)
          if [ "$n" -lt "$MAXPG" ] && [ "$(freemem $g)" -gt 4500 ] && [ "$n" -lt "$bn" ]; then best=$g; bn=$n; fi
        done; echo "$best"; }
for SEED in $SEEDS; do
  for V in $ARMS; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$SEED (done)" >> "$LOG"; continue; }
    G=""; while [ -z "$G" ]; do G=$(pick); [ -z "$G" ] && sleep 30; done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 45     # let it claim its memory before the next pick
  done
done
while [ "$(ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/ && /runs\/rank_matched/' | wc -l)" -gt 0 ]; do sleep 60; done
missing=0
for SEED in $SEEDS; do for V in $ARMS; do
  [ -f "$R/p0/${V}_s${SEED}/${V}.pt" ] || { echo "MISSING $V s$SEED" >> "$LOG"; missing=$((missing+1)); }
done; done
echo "$(date +%H:%M) missing=$missing" >> "$LOG"
[ "$missing" -eq 0 ] || { echo "not all checkpoints present -- evaluation NOT run" >> "$LOG"; exit 1; }
touch "$R/.train_done"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank at matched length: trained AND tested at T=1024" \
  --out "$REPO/RANK_MATCHED.md" >> "$LOG" 2>&1
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:0 \
  --check-json "$REPO/RANK_MATCHED.json" --out "$REPO/RANK_MATCHED_STRATA.json" >> "$LOG" 2>&1
python3 -u -m mapformer.eval_rank_strata --runs-dir "$REPO/runs/rank_sweep" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:1 \
  --check-json "$REPO/RANK_SWEEP.json" --out "$REPO/RANK_SWEEP_STRATA.json" >> "$LOG" 2>&1
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms $ARMS --device cuda:0 \
  --out "$REPO/RANK_MATCHED_GEOMETRY.md" >> "$LOG" 2>&1
touch "$REPO/.rank_matched_done"; echo "$(date) DONE" >> "$LOG"
