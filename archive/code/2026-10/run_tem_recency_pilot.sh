#!/usr/bin/env bash
# TEM_RECENCY_PILOT.md. Waits for the addition pilot 2 driver, then runs 5 TEM recency arms.
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/tem_recency_pilot; mkdir -p "$OUT/logs"; cd /home/prashr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
while ps -u "$USER" -o comm=,args= | awk '$1=="bash" && $2 !~ /^-/ && /run_addition_pilot2\.sh$/' | grep -q .; do sleep 60; done
g=0
for spec in "TEMRecency 0" "TEMRecency_Query 0" "TEMRecency_Query_Installed 0" "TEMRecency 1" "TEMRecency_Query 1"; do
  set -- $spec; v=$1; s=$2
  echo "$(date +%H:%M) $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" --k-max 64 --T 1024 \
    --eval-T 1024 2048 --epochs 300 --n-batches 48 --batch-size 16 --schedule cosine --lr 1e-3 \
    --n-layers 1 --d-model 128 --n-heads 2 --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
    > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  g=$((1-g)); sleep 5
done
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /tem_recency_pilot/' | grep -q .; do sleep 60; done
n=$(ls $OUT/*/*_recency.json 2>/dev/null | wc -l); echo "jsons=$n/5"; [ "$n" -eq 5 ] && touch "$OUT/.done"
echo "tem pilot finished $(date)"
