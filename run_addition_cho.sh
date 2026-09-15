#!/usr/bin/env bash
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/addition_cho; mkdir -p "$OUT/logs"; cd /home/prashr
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
g=0
for v in ChoCoupledAPE CoupledAPE; do
  setsid nohup python3 -u -m mapformer.train_addition --variant "$v" --fmt shared --seed 0 --dmax 30 \
    --epochs 500 --n-batches 100 --batch-size 1000 --lr 1e-4 --weight-decay 0.0 --warmup-frac 0.01 --max-pos 202 \
    --n-layers 1 --n-heads 4 --d-model 512 --eval-digits 30 60 100 150 200 --eval-examples 256 \
    --device "cuda:$g" --output-dir "$OUT/$v" > "$OUT/logs/$v.log" 2>&1 &
  echo "$(date +%H:%M) $v -> cuda:$g"; g=1; sleep 5
done
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_addition/ && /addition_cho/' | grep -q .; do sleep 60; done
n=$(ls $OUT/*/*_addition.json 2>/dev/null | wc -l); echo "jsons=$n/2"; [ "$n" -eq 2 ] && touch "$OUT/.done"
echo "cho repro finished $(date)"
