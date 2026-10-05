#!/usr/bin/env bash
# MapPoPE with pairwise frequencies on the paper torus. Pre-registration: MAPPOPE_PAIR_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/mappope_pair.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-8}"
drv_lock "$REPO/.run_mappope_pair.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/mappope_pair"; mkdir -p "$R/p0"
ARMS="Vanilla MapPoPE-Pair MapPoPE-Flat Vanilla_r4 MapPoPE-Pair_r4 MapPoPE_r4"
GUARD=(model.py model_rank.py model_pope.py model_pope_pair.py train.py train_variant.py train_pair.py eval_pair.py
       eval_noise_refine.py environment.py data_parallel.py analyze_mappope_pair.py stats_core.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
for S in $(seq 10 25); do for V in $ARMS; do
  case "$V" in *_r4) [ "$S" -gt 17 ] && continue;; esac          # rank-4 arms: seeds 10-17; rank-2 arms: 10-25
  OUT="$R/p0/${V}_s${S}"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_pair --variant "$V" --seed "$S" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --lr 1e-3 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/*.pt 2>/dev/null | wc -l); [ "$N" -eq 72 ] || drv_fail "only $N/72 checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
python3 -u -m mapformer.eval_pair --runs-dir "$R" --variants Vanilla MapPoPE-Pair MapPoPE-Flat --noises 0.0 --seeds $(seq 10 25) \
  --lengths 128 512 1024 --n-trials 100 --device cuda:0 --out "$REPO/MAPPOPE_PAIR_R2.md" \
  --title "MapPoPE pairwise frequencies, paper torus, rank 2" >> "$LOG" 2>&1 || drv_fail eval-r2
python3 -u -m mapformer.eval_pair --runs-dir "$R" --variants Vanilla_r4 MapPoPE-Pair_r4 MapPoPE_r4 --noises 0.0 --seeds $(seq 10 17) \
  --lengths 128 512 1024 --n-trials 100 --device cuda:0 --out "$REPO/MAPPOPE_PAIR_R4.md" \
  --title "MapPoPE pairwise frequencies, paper torus, rank 4" >> "$LOG" 2>&1 || drv_fail eval-r4
python3 -u -m mapformer.analyze_mappope_pair > "$REPO/MAPPOPE_PAIR_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.mappope_pair_done" "$REPO/MAPPOPE_PAIR_R2.json" "$REPO/MAPPOPE_PAIR_R4.json" "$REPO/MAPPOPE_PAIR_ANALYSIS.txt"
