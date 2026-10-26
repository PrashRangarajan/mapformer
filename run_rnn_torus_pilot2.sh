#!/usr/bin/env bash
# RNN_TORUS_PREREG.md, pilot 2 (before registration; seed 3391, OUTSIDE the batch, rule 29): full-length (900-epoch) runs
# of selected RNN arms at their pilot-1 rates, then the batch's evaluation path end to end on GPU (eval_rnn_torus
# noise_refine at T = 512 / 1024 / 2048, strata, own map, context destruction). Purpose: convergence behaviour within the
# budget, timing, memory, and an end-to-end check of the evaluation code on real trained RNNs. Not part of the batch.
# At most 2 of these jobs per GPU. Usage:
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" PILOT2_JOBS="LSTM_L1_h157:1:3e-3 LSTM_L1_h313:1:3e-3" \
#     setsid nohup bash run_rnn_torus_pilot2.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rnn_torus_pilot2.log"; R="$REPO/runs/rnn_torus_pilot/full"
REQ_GPUS="${DRV_GPUS:-}"; JOBS_STR="${PILOT2_JOBS:-}"
source "$REPO/lib_driver.sh"
[ -n "$REQ_GPUS" ] && [ -n "$JOBS_STR" ] || { _drv_log "REFUSED: set DRV_GPUS and PILOT2_JOBS"; exit 1; }
DRV_GPUS="$REQ_GPUS"; export DRV_GPUS; DRV_MAXPG=2; DRV_MINFREE=4000; DRV_SPACING=20
export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rnn_torus_pilot2.lock" || exit 1
S=3391; read -r -a JOBS <<< "$JOBS_STR"
cd "$REPO/.."
echo "start $(date) code $(cd "$REPO" && git rev-parse HEAD) jobs $JOBS_STR" >> "$LOG"
for J in "${JOBS[@]}"; do
  IFS=: read -r V NL LR <<< "$J"; ROOT="lr$(python3 -c "print(f'{float(\"$LR\"):.0e}')")"; OUT="$R/$ROOT/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $V (done)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$V L$NL lr $LR s$S -> cuda:$G"
  OMP_NUM_THREADS=3 drv_launch "$R/$ROOT/${V}_s$S.log" python3 -u -m mapformer.train_rnn --variant "$V" --seed "$S" \
    --lr "$LR" --epochs 900 --n-batches 98 --batch-size 16 --n-steps 1024 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --data-workers 3 --n-layers "$NL" --device "cuda:$G" --output-dir "$OUT"
done
sleep 60; DRV_WAIT_POLL=60 drv_wait_dir "$R/"
G=$(drv_wait_slot)
for J in "${JOBS[@]}"; do
  IFS=: read -r V NL LR <<< "$J"; ROOT="lr$(python3 -c "print(f'{float(\"$LR\"):.0e}')")"
  [ -f "$R/$ROOT/p0/${V}_s$S/$V.pt" ] || { _drv_log "MISSING $V"; continue; }
  python3 -u -m mapformer.eval_rnn_torus noise_refine --runs-dir "$R/$ROOT" --variants "$V" --noises 0.0 --seeds $S \
    --lengths 512 1024 2048 --n-trials 100 --device "cuda:$G" --title "pilot2 $V" --out "$R/EVAL_${V}.md" >> "$LOG" 2>&1
  python3 -u -m mapformer.eval_rnn_torus strata --runs-dir "$R/$ROOT" --variants "$V" --seeds $S --lengths 1024 \
    --n-trials 100 --device "cuda:$G" --check-json "$R/EVAL_${V}.json" --out "$R/STRATA_${V}.json" >> "$LOG" 2>&1
  python3 -u -m mapformer.eval_rnn_torus noise_refine --runs-dir "$R/$ROOT" --variants "$V" --noises 0.0 --seeds $S \
    --env-seed $S --lengths 1024 --n-trials 100 --device "cuda:$G" --title "pilot2 own $V" --out "$R/OWN_${V}.md" >> "$LOG" 2>&1
  python3 -u -c "
import torch, json
from mapformer import analyze_rnn_torus as AR
from mapformer.model_rnn import ARM_SPECS
cells = {'$V': ('$ROOT', '$V', ARM_SPECS['$V'][1], float('$LR'), False)}
AR.run_cd(torch.device('cuda:$G'), cells=cells, runs='$R', seeds=[$S], out='$R/CD_${V}.json')
" >> "$LOG" 2>&1
done
touch "$REPO/runs/rnn_torus_pilot/.full_done"; _drv_log "pilot2 done"
