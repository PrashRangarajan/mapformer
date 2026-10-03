#!/usr/bin/env bash
# Leak remedies on the new-object task. Pre-registration: LEAK_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/leak.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-15}"
drv_lock "$REPO/.run_leak.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/leak"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_codes.py train.py train_variant.py train_newobj.py environment_nd.py \
  environment_newobj.py data_parallel.py leak_eval.py analyze_leak.py stats_core.py || exit 1
for S in 0 1 2 3 4 5 6 7; do for ARM in MapWM ActOnly NormStep; do
  OUT="$R/p0/${ARM}_s${S}"; [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $ARM s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_newobj --variant "$ARM" --seed "$S" \
    --epochs 900 --n-steps 1024 --batch-size 16 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq 24 ] || drv_fail "only $N/24 eval.json"
python3 -u -m mapformer.analyze_leak > "$REPO/LEAK_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.leak_done" "$REPO/LEAK_ANALYSIS.txt" "$REPO/LEAK.json"
