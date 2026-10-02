#!/usr/bin/env bash
# Dimension x wrap share at rank D+1. Pre-registration: RANK_WRAP_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_wrap.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-15}"
drv_lock "$REPO/.run_rank_wrap.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/rank_wrap"; mkdir -p "$R"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment_nd.py \
  data_parallel.py eval_nd.py analyze_rank_nd_secondary.py analyze_rank_wrap.py stats_core.py || exit 1
CELLS=("2 32 Vanilla_r3ph" "2 10 Vanilla_r3ph" "3 18 Vanilla_r4ph" "3 10 Vanilla_r4ph")
for S in 0 1 2 3 4 5 6 7; do for c in "${CELLS[@]}"; do set -- $c
  OUT="$R/N$2/D$1/${3}_s${S}"; [ -f "$OUT/$3.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) D$1 N$2 $3 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$R/N$2_D$1_s${S}.log" python3 -u -m mapformer.train_variant --variant "$3" \
    --env nd --n-dims "$1" --grid-size "$2" --seed "$S" --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 \
    --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state
done; done
drv_wait_dir "$R/"
REQ=(); for S in 0 1 2 3 4 5 6 7; do for c in "${CELLS[@]}"; do set -- $c; REQ+=("$R/N$2/D$1/${3}_s${S}/$3.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
for c in "${CELLS[@]}"; do set -- $c
  python3 -u -m mapformer.eval_nd --runs-dir "$R/N$2" --configs "$1:$2:$3" --lengths 1024 2048 --n-trials 100 \
    --device cuda:0 --out "$R/N$2/EVAL_D$1.md" >> "$LOG" 2>&1 || drv_fail "eval D$1 N$2"
done
python3 -u -m mapformer.analyze_rank_wrap > "$REPO/RANK_WRAP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_wrap_done" "$REPO/RANK_WRAP_ANALYSIS.txt" "$REPO/RANK_WRAP.json"
