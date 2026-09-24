#!/usr/bin/env bash
# Warm-start stability test. Pre-registration: RANK_PROJ_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_proj.log"
exec 9>"$REPO/.run_rank_proj.lock"
flock -n 9 || { echo "$(date) REFUSED: another run_rank_proj.sh holds the lock" >> "$LOG"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
echo "start $(date)" >> "$LOG"
fail(){ echo "$(date) FAILED: $1 -- done marker NOT set" >> "$LOG"; exit 1; }
P="$REPO/runs/rank_proj"; R="$REPO/runs/rank_proj_train"; mkdir -p "$R/p0"
SEEDS="0 1 2 3 4 5 6 7"
ntrain(){ ps -u "$USER" -o comm=,args= | awk -v d="--device cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }
freemem(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
pick(){ local best="" bn=99; for g in 0 1; do n=$(ntrain $g)
          if [ "$n" -lt 5 ] && [ "$(freemem $g)" -gt 4500 ] && [ "$n" -lt "$bn" ]; then best=$g; bn=$n; fi
        done; echo "$best"; }
# 1. FROZEN: existence on all 8 seeds (eval only)
python3 -u -m mapformer.eval_noise_refine --runs-dir "$P" --variants Vanilla --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "Projected r=2 (from solved r=4), frozen" --out "$REPO/RANK_PROJ_FROZEN.md" >> "$LOG" 2>&1 || fail frozen
# 2. TRAINABLE: the continuation's exact recipe from the projected weights
for SEED in $SEEDS; do
  OUT="$R/p0/Vanilla_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/Vanilla.pt" ] && { echo "skip s$SEED (done)" >> "$LOG"; continue; }
  G=""; while [ -z "$G" ]; do G=$(pick); [ -z "$G" ] && sleep 30; done
  echo "$(date +%H:%M:%S) Vanilla(proj) s$SEED -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_variant --variant Vanilla --seed "$SEED" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
    --init-from "$P/p0/Vanilla_s${SEED}/Vanilla.pt" --data-seed-offset 1 --save-full-state \
    > "$R/Vanilla_s${SEED}.log" 2>&1 &
  sleep 45
done
while [ "$(ps -u "$USER" -o comm=,args= | awk -v r="$R/" '$1=="python3" && /mapformer\.train_variant/ && index($0,r)' | wc -l)" -gt 0 ]; do sleep 60; done
missing=0; for SEED in $SEEDS; do [ -f "$R/p0/Vanilla_s${SEED}/Vanilla.pt" ] || { echo "MISSING s$SEED" >> "$LOG"; missing=$((missing+1)); }; done
[ "$missing" -eq 0 ] || fail "missing=$missing"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "Projected r=2, trained 900 epochs (continuation recipe)" --out "$REPO/RANK_PROJ_TRAIN.md" >> "$LOG" 2>&1 || fail train_eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla --seeds $SEEDS --lengths 1024 \
  --n-trials 100 --device cuda:1 --check-json "$REPO/RANK_PROJ_TRAIN.json" --out "$REPO/RANK_PROJ_TRAIN_STRATA.json" >> "$LOG" 2>&1 || fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla --seeds $SEEDS --device cuda:1 \
  --out "$REPO/RANK_PROJ_TRAIN_GEOMETRY.md" >> "$LOG" 2>&1 || fail geometry
# 3. the comparison needs the continuation's control arm
until [ -f "$REPO/.rank_matched_e900c_done" ]; do sleep 120; done
python3 -u -m mapformer.analyze_rank_proj > "$REPO/RANK_PROJ_ANALYSIS.txt" 2>&1 || fail analyze
touch "$REPO/.rank_proj_done"; echo "$(date) DONE" >> "$LOG"
