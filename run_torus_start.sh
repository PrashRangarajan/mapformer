#!/usr/bin/env bash
# TORUS_START_PREREG.md batch: 5 arms x seeds 1200-1207 on the torus. Waits for the ACCOUNTS batch's marker first.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_torus_start.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/torus_start.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=4000; DRV_SPACING=20    # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
drv_lock "$REPO/.run_torus_start.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/torus_start/p0"; mkdir -p "$R"
SEEDS=(1200 1201 1202 1203 1204 1205 1206 1207); ARMS=(SO4 AO4 AO4_high SO4_high AO4_mid)
GUARD=(train_torus_start.py model_torus_start.py model_wrap_exist.py model_sign.py model.py lie_groups.py train.py environment.py
       data_parallel.py eval_noise_refine.py eval_rank_strata.py train_variant.py analyze_torus_start.py stats_core.py lib_driver.sh
       run_torus_start.sh)
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
drv_md5_guard "$R" "${GUARD[@]}" || exit 1                    # Amendment 1: pinned BEFORE the wait
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; waiting for .accounts_done"
while [ ! -f "$REPO/.accounts_done" ]; do
  grep -q "FAILED\|ABORT\|REFUSED" "$REPO/accounts.log" && { _drv_log "ACCOUNTS failed; starting anyway after its jobs exit"; break; }
  sleep 120
done
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /runs\/accounts\//' | grep -q .; do sleep 60; done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed during the wait"
_drv_log "ACCOUNTS finished; launching"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do
  OUT="$R/${A}_s$S"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_torus_start --arm "$A" --seed "$S" --device "cuda:$G" \
    --output-dir "$OUT"
done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do REQ+=("$R/${A}_s$S/eval.json" "$R/${A}_s$S/$A.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 80 ] || drv_fail "expected 80 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_torus_start > "$REPO/TORUS_START_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.torus_start_done" "$REPO/TORUS_START_ANALYSIS.txt" "$REPO/TORUS_START_VERDICTS.json"
