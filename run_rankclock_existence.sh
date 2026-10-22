#!/usr/bin/env bash
# RANKCLOCK_PREREG.md Sec. 3: rule-14 existence (frozen + trainable) before the batch. Waits for the r4 pilot (seed 1990),
# builds each class's best construction from it, warm-starts each for 60 cosine epochs at peak lr 1e-3 with the batch recipe
# (GPU 1, two at a time), and scores held vs lost. Pilot seed 1990 = outside the batch.
#   cd /home/prashr/mapformer && setsid nohup bash run_rankclock_existence.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rankclock_existence.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=2; DRV_MINFREE=4500; DRV_SPACING=20
DRV_GPUS=1; export DRV_GPUS; export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rankclock_existence.lock" || exit 1
P="$REPO/runs/rankclock_pilot"; SRC="$P/Vanilla_r4mi_s1990/Vanilla_r4mi.pt"
echo "start $(date)" >> "$LOG"
until [ -f "$SRC" ]; do sleep 60; done
_drv_log "source ready: $SRC"
cd "$REPO/.."
python3 -u -m mapformer.construct_rankclock --device cuda:1 --n-trials 60 --warmstart-dir "$P/warmstart" >> "$LOG" 2>&1 \
  || drv_fail construct
for C in r2:Vanilla slot:Vanilla_r2clock slotxy:Vanilla_r2clockxy AO:Vanilla_r2AO DF:Vanilla_r2DF; do
  CL=${C%%:*}; V=${C#*:}; OUT="$P/warm_$CL"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip warm $CL (done)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "warm $CL -> cuda:$G"
  OMP_NUM_THREADS=4 drv_launch "$OUT.log" python3 -u -m mapformer.train_variant --variant "$V" --seed 1990 \
    --init-from "$P/warmstart/$CL/$V.pt" --epochs 60 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --device "cuda:$G" \
    --output-dir "$OUT"
done
sleep 30
DRV_WAIT_POLL=30 drv_wait_dir "$P/warm_"
python3 -u -m mapformer.construct_rankclock --device cuda:1 --n-trials 60 --score-warm "$P" >> "$LOG" 2>&1 || drv_fail score
drv_done "$P/.existence_done" "$P/construct_rankclock.json"
