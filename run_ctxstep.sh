#!/usr/bin/env bash
# Context-dependent step, registered batch. Pre-registration: CTXSTEP_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/ctxstep.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-15}"
drv_lock "$REPO/.run_ctxstep.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/ctxstep"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_context_step.py model_selective.py model_rank.py model_baseline_rope.py \
  train.py train_variant.py train_ctxstep3.py environment.py environment_textworld.py environment_textworld_ctx3.py \
  data_parallel.py docs/audits/2026-09-27/swap_test.py || exit 1
CELLS=()
for CUE in lead trail; do
  CELLS+=("$CUE far HSR 2" "$CUE far CF 1" "$CUE far CG 1" "$CUE far SR 1" "$CUE near CG 1" "$CUE near SR 1")
done
for S in 6 7 8 9 10 11 12 13; do for c in "${CELLS[@]}"; do set -- $c
  OUT="$R/p0/${1}_${2}_${3}_s${S}"; [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $1 $2 $3 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_ctxstep3 --variant "$3" --n-layers "$4" \
    --cue "$1" --dist "$2" --seed "$S" --epochs 1800 --n-steps 2048 --batch-size 8 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq 96 ] || drv_fail "only $N/96 eval.json"
SW="$REPO/CTXSTEP_SWAP.jsonl"; : > "$SW"
for d in "$R"/p0/*_s*/; do b=$(basename "$d"); set -- $(echo "$b" | tr '_' ' ')
  L=1; [ "$3" = HSR ] && L=2
  PYTHONPATH=/home/prashr python3 "$REPO/docs/audits/2026-09-27/swap_test.py" --task ctx3 --cue "$1" --dist "$2" \
    --ckpt "$d$3.pt" --arm "$3" --layers $L --offset 15 --json "$SW" >> "$LOG" 2>&1 || drv_fail "swap $b"
done
python3 -u -m mapformer.analyze_ctxstep > "$REPO/CTXSTEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.ctxstep_done" "$REPO/CTXSTEP_SWAP.jsonl" "$REPO/CTXSTEP_ANALYSIS.txt"
