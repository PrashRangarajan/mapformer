#!/usr/bin/env bash
# ADDITION_PILOT2.md: faithful coupled-APE control, dmax 30, eval to 120 digits. One seed.
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/addition_pilot2; mkdir -p "$OUT/logs"; cd /home/prashr
MAXPG=5
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_addition/' | wc -l; }
launch(){ v=$1; fmt=$2; L=$3
  d="$OUT/${v}_${fmt}_L${L}"; [ -f "$d/${v}_addition.json" ] && return
  while :; do a=$(on_gpu 0); b=$(on_gpu 1)
    if [ "$a" -le "$b" ] && [ "$a" -lt $MAXPG ]; then g=0; break; fi
    if [ "$b" -lt $MAXPG ]; then g=1; break; fi; sleep 15; done
  echo "$(date +%H:%M) $v $fmt L$L -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_addition --variant "$v" --fmt "$fmt" --n-layers "$L" \
    --seed 0 --dmax 30 --epochs 200 --n-batches 100 --batch-size 512 --lr 1e-3 --d-model 256 --n-heads 4 \
    --eval-digits 16 30 45 60 90 120 --eval-examples 256 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_${fmt}_L${L}.log" 2>&1 &
  sleep 5; }
for L in 1 2; do
  for v in CoupledAPE Vanilla_r4 RoPE CoupledRoPE Abs_r4 NoPE; do launch $v role $L; done
  for v in CoupledAPE Vanilla_r4 RoPE; do launch $v shared $L; done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
n=$(ls $OUT/*/*_addition.json 2>/dev/null | wc -l); echo "jsons=$n/18"
[ "$n" -eq 18 ] && touch "$OUT/.done"
echo "pilot2 finished $(date)"
