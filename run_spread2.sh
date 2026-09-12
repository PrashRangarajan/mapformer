#!/usr/bin/env bash
# SPREAD2_PREREG.md: the exposure test, OFF THE CEILING.
# SPREAD's exposure-matched cells both sat at exactly 1.000, so the contrast could not fire.
# Calibration (runs/spread_cal): m4 scores 0.605 / 0.697 / 0.984 at 20 / 40 / 80 epochs, so 60
# is targeted to land near 0.8. Exposure per token = epochs * 5376 / m, so m16 needs 240 epochs
# to match m4 at 60.
#
# Gated behind the paper-task rerun so the three batches run in sequence.
# The wait loops count ONLY this batch's runs -- SPREAD's counted every train_recency process,
# which left the calibration driver idling behind an unrelated batch.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/spread2
mkdir -p "$OUT/logs"
K4="1,4,16,64"
K16="1,2,3,4,6,8,11,16,22,26,32,38,45,52,58,64"
MAXPG=4
mine(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /runs\/spread2/' | wc -l; }
others(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/ && !/runs\/spread2/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /runs\/spread2/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local arm=$1 ep=$2 s=$3; shift 3
  [ -f "$OUT/${arm}_s${s}/VanillaEM_P0_r4_recency.json" ] && { echo "skip $arm s$s"; return; }
  while [ "$(mine)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $arm s$s (${ep} ep) -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant VanillaEM_P0_r4 --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs "$ep" --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${arm}_s${s}" "$@" \
    > "$OUT/logs/${arm}_s${s}.log" 2>&1 &
  sleep 6; }

echo "$(date +%H:%M) waiting for the paper-task rerun"
until [ -f "$REPO/runs/paper_task_rerun/.done" ] || [ "$(others)" -eq 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs free; starting"

# long arm first so it is not the tail; seed outer within each block
for s in $(seq 0 7); do go m16_e240 240 "$s" --k-set "$K16"; done
for s in $(seq 0 7); do
  go m4_e60  60 "$s" --k-set "$K4"
  go m16_e60 60 "$s" --k-set "$K16"
  go m64_e60 60 "$s"
done
while [ "$(mine)" -gt 0 ]; do sleep 30; done
missing=0
for arm in m4_e60 m16_e60 m16_e240 m64_e60; do for s in $(seq 0 7); do
  [ -f "$OUT/${arm}_s${s}/VanillaEM_P0_r4_recency.json" ] || { echo "MISSING $arm s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"
