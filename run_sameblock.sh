#!/usr/bin/env bash
# SAMEBLOCK_PREREG.md
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/sameblock; mkdir -p "$OUT/logs"; cd /home/prashr
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
MAXPG=3
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_addition/ && /sameblock/' | wc -l; }
launch(){ v=$1; fmt=$2; s=$3
  d="$OUT/${v}_${fmt}_s${s}"; [ -f "$d/${v}_addition.json" ] && return
  while :; do a=$(on_gpu 0); b=$(on_gpu 1)
    if [ "$a" -le "$b" ] && [ "$a" -lt $MAXPG ]; then g=0; break; fi
    if [ "$b" -lt $MAXPG ]; then g=1; break; fi; sleep 30; done
  echo "$(date +%H:%M) $v $fmt s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_addition --variant "$v" --fmt "$fmt" --seed "$s" --dmax 30 \
    --epochs 500 --n-batches 100 --batch-size 1000 --lr 1e-4 --weight-decay 0.0 --warmup-frac 0.01 --max-pos 202 \
    --n-layers 1 --n-heads 4 --d-model 512 --amp --eval-digits 30 60 100 150 --eval-examples 512 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_${fmt}_s${s}.log" 2>&1 &
  sleep 10; }
# seed 0 of everything first, so a complete one-seed table lands early
for v in ChoPos_signed ChoPos_abs ChoPos_rope ChoPos_coupled ChoPos_nope; do launch $v role 0; done
launch ChoPos_coupled shared 0; launch ChoPos_signed shared 0
for s in 1 2; do for v in ChoPos_signed ChoPos_abs ChoPos_rope ChoPos_coupled; do launch $v role $s; done; done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
n=$(ls $OUT/*/*_addition.json 2>/dev/null | wc -l); echo "jsons=$n/15"; [ "$n" -eq 15 ] && touch "$OUT/.done"
echo "sameblock finished $(date)"
