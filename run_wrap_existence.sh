#!/usr/bin/env bash
# WRAP_EXISTENCE_PREREG.md batch: seeds 600-607. Phase 1: SO and AO from scratch; phase 2 (per seed, once SO's checkpoint
# exists): SO_cont and AO_wrap from it. GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_wrap_existence.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/wrap_existence.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27); ring jobs peak ~1.25 GB
export DRV_GPUS="${DRV_GPUS:-0}"
drv_lock "$REPO/.run_wrap_existence.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/wrap_existence/p0"; mkdir -p "$R"
SEEDS=(600 601 602 603 604 605 606 607)
GUARD=(train_wrap_exist.py model_wrap_exist.py model_sign.py train_cancel.py environment_cancel.py environment_nd.py model.py
       train.py analyze_wrap_existence.py stats_core.py lib_driver.sh run_wrap_existence.sh data_parallel.py environment.py
       rescore_hook.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$REPO/runs/wrap_existence" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
launch(){  # arm seed [init]
  local A="$1" S="$2" OUT="$R/${1}_s${2}" G; shift 2
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; return 0; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_wrap_exist --arm "$A" --seed "$S" "$@" \
    --device "cuda:$G" --output-dir "$OUT"
}
for S in "${SEEDS[@]}"; do for A in SO AO; do launch "$A" "$S"; done; done
for S in "${SEEDS[@]}"; do
  while [ ! -f "$R/SO_s$S/eval.json" ]; do
    running "$R/SO_s$S" || { [ -f "$R/SO_s$S/eval.json" ] || drv_fail "SO s$S ended without eval.json"; }
    sleep 60
  done
  for A in SO_cont AO_wrap; do launch "$A" "$S" --init "$R/SO_s$S/SO.pt"; done
done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for A in SO AO SO_cont AO_wrap; do REQ+=("$R/${A}_s${S}/eval.json" "$R/${A}_s${S}/${A}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
drv_md5_guard "$REPO/runs/wrap_existence" "${GUARD[@]}" || drv_fail "code changed before the analysis"
[ "${#REQ[@]}" -eq 64 ] || drv_fail "expected 64 artifacts"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_wrap_existence > "$REPO/WRAP_EXISTENCE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.wrap_existence_done" "$REPO/WRAP_EXISTENCE_ANALYSIS.txt" "$REPO/WRAP_EXISTENCE_VERDICTS.json"
