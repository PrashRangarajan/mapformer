#!/usr/bin/env bash
# SIGNSEP pilots, wave 2 (Amendment 2: tau exempt from weight decay, the registered default): the abs_wrap pilots of
# run_signsep_pilot.sh rerun on the SAME seeds, so each pairs with its tau-decay-on run.
# SIGNSEP pilots (SIGNSEP_PREREG.md Sec. 9): seeds 2380-2387, OUTSIDE the registered batch (2300-2307), never pooled.
# Each new arm from scratch at both rates, the rule-14 trainable warm starts (F also at its own start scale, Amendment 1
# finding 5), and the signed anchor at 3e-3 (G0 risk).
# One GPU beside the other user's jobs, at most 4 of ours on it (the user's limit).
#   cd /home/prashr/mapformer && setsid nohup bash run_signsep_pilot.sh > /dev/null 2>&1 &
# Needs runs/signsep_pilot/construct/*_construct.pt first: python3 -m mapformer.construct_signsep --save <that dir>
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/signsep_pilot2.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=3000; DRV_SPACING=20            # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-1}"
drv_lock "$REPO/.run_signsep_pilot3.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/signsep_pilot2"; C="$REPO/runs/signsep_pilot/construct"; mkdir -p "$R"
drv_require "$C/abs_wrap_construct.pt" "$C/signed_frozen_construct.pt" "$C/signed_frozen_startc_s2384_construct.pt" || { _drv_log "ABORT: no constructions"; exit 1; }
drv_md5_guard "$R" model_signsep.py train_signsep.py construct_signsep.py model_signroute.py model_wrap_exist.py \
  train_wrap_mech.py train.py train_cancel.py environment_steps.py environment_cancel.py model.py model_sign.py \
  data_parallel.py lib_driver.sh \
  run_signsep_pilot2.sh || exit 1
_drv_log "pilot start; code $(cd "$REPO" && git rev-parse HEAD) (pilots may run on an uncommitted tree); GPUs $DRV_GPUS"
# name arm lr seed [init]
JOBS=(
  "abs_wrap_lr1e-3_s2380 abs_wrap 1e-3 2380"
  "abs_wrap_past_lr1e-3_s2382 abs_wrap_past 1e-3 2382"
  "abs_wrap_warm_lr1e-3_s2383 abs_wrap 1e-3 2383 $C/abs_wrap_construct.pt"
  "abs_wrap_lr3e-3_s2386 abs_wrap 3e-3 2386"
)
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
REQ=()
for J in "${JOBS[@]}"; do
  read -r NAME A LR S INIT <<< "$J"
  OUT="$R/$NAME"; REQ+=("$OUT/eval.json")
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $NAME (done)"; continue; }
  running "$OUT" && { _drv_log "skip $NAME (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$NAME -> cuda:$G"
  EXTRA=(); [ -n "${INIT:-}" ] && EXTRA=(--init-from "$INIT")
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_signsep --arm "$A" --lr "$LR" --seed "$S" \
    --device "cuda:$G" --output-dir "$OUT" "${EXTRA[@]}"
done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
drv_require "${REQ[@]}" || drv_fail "missing pilot runs"
drv_md5_guard "$R" model_signsep.py train_signsep.py construct_signsep.py model_signroute.py model_wrap_exist.py \
  train_wrap_mech.py train.py train_cancel.py environment_steps.py environment_cancel.py model.py model_sign.py \
  data_parallel.py lib_driver.sh \
  run_signsep_pilot2.sh || drv_fail "code changed during the pilot"
drv_done "$REPO/.signsep_pilot2_done" "${REQ[@]}"
