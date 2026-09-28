#!/usr/bin/env bash
# H3, the cancellation knob. Pre-registration: CANCEL_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/cancel.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-1}"; DRV_SPACING="${DRV_SPACING:-10}"
DRV_MODULE_RE="mapformer[.]train_cancel"      # one H3 job per GPU beside the train_variant batches
drv_lock "$REPO/.run_cancel.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/cancel"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_baseline_rope.py train.py train_variant.py environment_nd.py environment_cancel.py train_cancel.py data_parallel.py || exit 1
CELLS=(); for P in 0.5 0.75 0.9 1.0; do CELLS+=("Vanilla 1 $P" "RoPE 1 $P" "RoPE 2 $P" "RoPE 3 $P"); done
for S in 0 1 2 3 4 5 6 7; do for c in "${CELLS[@]}"; do set -- $c
  OUT="$R/p0/${1}_L${2}_p${3}_s${S}"
  [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $1 L$2 p$3 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_cancel --variant "$1" --n-layers "$2" \
    --p-plus "$3" --seed "$S" --epochs 300 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq 128 ] || drv_fail "only $N/128 eval.json"
python3 -u -m mapformer.analyze_cancel > "$REPO/CANCEL_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.cancel_done" "$REPO/CANCEL_ANALYSIS.txt"
