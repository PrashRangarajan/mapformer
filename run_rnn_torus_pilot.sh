#!/usr/bin/env bash
# RNN_TORUS_PREREG.md, pilot (before registration; seeds OUTSIDE the batch, rule 29): learning rate per RNN arm.
# Each run uses the batch's exact recipe and its 900-epoch cosine schedule, and is STOPPED at epoch $STOP (default 300) by
# PID: the logged training loss over epochs 271-300 is the pilot readout (selection rule in the pre-registration).
# At most 2 of these jobs per GPU (DRV_MAXPG=2); GPUs in DRV_GPUS (required). Seed 3390.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_rnn_torus_pilot.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rnn_torus_pilot.log"; R="$REPO/runs/rnn_torus_pilot/lr"
REQ_GPUS="${DRV_GPUS:-}"
source "$REPO/lib_driver.sh"
[ -n "$REQ_GPUS" ] || { _drv_log "REFUSED: set DRV_GPUS"; exit 1; }
DRV_GPUS="$REQ_GPUS"; export DRV_GPUS; DRV_MAXPG=2; DRV_MINFREE=4000; DRV_SPACING=20
export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rnn_torus_pilot.lock" || exit 1
STOP="${STOP:-300}"; S=3390
mkdir -p "$R"; cd "$REPO/.."
echo "start $(date) code $(cd "$REPO" && git rev-parse HEAD)" >> "$LOG"
# arm:layers:lr -- longest first
JOBS=(LSTM_L1_h313:1:1e-3 LSTM_L1_h313:1:3e-3
      LSTM_L1_h157:1:1e-3 LSTM_L1_h157:1:3e-3 GRU_L1_h181:1:1e-3 GRU_L1_h181:1:3e-3
      LSTM_L1_h157:1:3e-4 LSTM_L1_h157:1:1e-2
      LSTM_L2_h111:2:1e-3 LSTM_L2_h111:2:3e-3
      GRU_L2_h128:2:3e-4 GRU_L2_h128:2:1e-3 GRU_L2_h128:2:3e-3 GRU_L2_h128:2:1e-2)
declare -A PIDS
watch_kill() {   # stop every launched run that has logged epoch $STOP
  local k
  for k in "${!PIDS[@]}"; do
    if grep -q "Epoch *$STOP/900" "$R/$k.log" 2>/dev/null; then
      kill "${PIDS[$k]}" 2>/dev/null && _drv_log "stopped $k at epoch $STOP (pid ${PIDS[$k]})"; unset "PIDS[$k]"
    elif ! kill -0 "${PIDS[$k]}" 2>/dev/null; then
      _drv_log "EXITED early $k (pid ${PIDS[$k]})"; unset "PIDS[$k]"
    fi
  done
}
for J in "${JOBS[@]}"; do
  IFS=: read -r V NL LR <<< "$J"; K="${V}_lr${LR}_s$S"
  while :; do watch_kill; G=$(drv_pick); [ -n "$G" ] && break; sleep 15; done
  OUT="$R/$K"; mkdir -p "$OUT"
  OMP_NUM_THREADS=3 setsid nohup python3 -u -m mapformer.train_rnn --variant "$V" --seed "$S" --epochs 900 --lr "$LR" \
    --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers "$NL" --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --data-workers 3 --device "cuda:$G" --output-dir "$OUT" > "$R/$K.log" 2>&1 &
  PIDS[$K]=$!; _drv_log "launched $K pid $! cuda:$G"; sleep "$DRV_SPACING"
done
while [ "${#PIDS[@]}" -gt 0 ]; do watch_kill; sleep 15; done
touch "$REPO/runs/rnn_torus_pilot/.lr_done"; _drv_log "pilot done"
