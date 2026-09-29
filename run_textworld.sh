#!/usr/bin/env bash
# Navigation told in words. Pre-registration: TEXTWORLD_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/textworld.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-10}"
DRV_MODULE_RE="mapformer[.]train_(textworld|variant)"
drv_lock "$REPO/.run_textworld.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/textworld"; mkdir -p "$R/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_baseline_rope.py train.py train_variant.py environment.py environment_textworld.py train_textworld.py data_parallel.py || exit 1
CELLS=("Vanilla_r4 1" "RoPE 1" "RoPE 2")
for S in 0 1 2 3 4 5 6 7; do for c in "${CELLS[@]}"; do set -- $c
  OUT="$R/p0/${1}_L${2}_s${S}"
  [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $1 L$2 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_textworld --variant "$1" --n-layers "$2" \
    --seed "$S" --epochs 900 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq 24 ] || drv_fail "only $N/24 eval.json"
python3 -u -m mapformer.probe_textworld --runs-dir "$R/p0" --out "$REPO/TEXTWORLD_PROBE.json" >> "$LOG" 2>&1 || drv_fail probe
python3 -u -m mapformer.analyze_textworld > "$REPO/TEXTWORLD_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.textworld_done" "$REPO/TEXTWORLD_PROBE.json" "$REPO/TEXTWORLD_ANALYSIS.txt"
