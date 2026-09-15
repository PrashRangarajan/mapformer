#!/usr/bin/env bash
# SAMEBLOCK amendment 1: trimmed queue on the validated fast code.
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/sameblock; cd /home/prashr
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
g=0
launch(){ v=$1; s=$2
  d="$OUT/${v}_role_s${s}"; [ -f "$d/${v}_addition.json" ] && return
  echo "$(date +%H:%M) $v role s$s -> cuda:$g (compile)"
  setsid nohup python3 -u -m mapformer.train_addition --variant "$v" --fmt role --seed "$s" --dmax 30 \
    --epochs 500 --n-batches 100 --batch-size 1000 --lr 1e-4 --weight-decay 0.0 --warmup-frac 0.01 --max-pos 202 \
    --n-layers 1 --n-heads 4 --d-model 512 --amp --compile --eval-digits 30 60 100 150 --eval-examples 512 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_role_s${s}.log" 2>&1 &
  g=$((1-g)); sleep 10; }
for s in 1 2; do launch ChoPos_signed $s; launch ChoPos_coupled $s; done
launch ChoPos_abs 1; launch ChoPos_rope 1
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_addition/ && /sameblock/' | grep -q .; do sleep 60; done
n=$(ls $OUT/*/*_addition.json 2>/dev/null | wc -l); echo "jsons=$n/12"; [ "$n" -eq 12 ] && touch "$OUT/.done"
echo "sameblock (amended) finished $(date)"
