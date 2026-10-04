#!/usr/bin/env bash
# NormStep on the text world. Pre-registration: TW_NORMSTEP_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/tw_normstep.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-3}"; DRV_SPACING="${DRV_SPACING:-10}"
drv_lock "$REPO/.run_tw_normstep.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/tw_normstep"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_codes.py model_textstep.py train.py train_variant.py environment.py \
  environment_textworld.py train_textworld.py train_tw_normstep.py data_parallel.py tw_normstep_readouts.py \
  analyze_tw_normstep.py stats_core.py || exit 1
for S in 10 11 12 13 14 15 16 17; do for A in MapWM NormStep NormStepNB DirOnly; do
  OUT="$R/p0/${A}_s${S}"
  [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $A s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_tw_normstep --arm "$A" --seed "$S" \
    --epochs 900 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq 32 ] || drv_fail "only $N/32 eval.json"
drv_md5_guard "$R" model.py model_rank.py model_codes.py model_textstep.py train.py train_variant.py environment.py \
  environment_textworld.py train_textworld.py train_tw_normstep.py data_parallel.py tw_normstep_readouts.py \
  analyze_tw_normstep.py stats_core.py || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_tw_normstep --readouts > "$REPO/TW_NORMSTEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.tw_normstep_done" "$REPO/TW_NORMSTEP.json" "$REPO/TW_NORMSTEP_ANALYSIS.txt"
