#!/usr/bin/env bash
# ADDITION_DESIGN.md pilot: 5 arms x 2 formats x {1,2} layers, seed 0. Learnability and range only.
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/addition_pilot; mkdir -p "$OUT/logs"; cd /home/prashr
MAXPG=5
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_addition/' | wc -l; }
for L in 1 2; do for fmt in role shared; do for v in Vanilla_r4 Abs_r4 RoPE NoPE CoupledRoPE; do
  d="$OUT/${v}_${fmt}_L${L}"; [ -f "$d/${v}_addition.json" ] && continue
  while :; do a=$(on_gpu 0); b=$(on_gpu 1)
    if [ "$a" -le "$b" ] && [ "$a" -lt $MAXPG ]; then g=0; break; fi
    if [ "$b" -lt $MAXPG ]; then g=1; break; fi; sleep 15; done
  echo "$(date +%H:%M) $v $fmt L$L -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_addition --variant "$v" --fmt "$fmt" --n-layers "$L" \
    --seed 0 --epochs 100 --n-batches 100 --batch-size 256 --lr 1e-3 --d-model 256 --n-heads 4 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_${fmt}_L${L}.log" 2>&1 &
  sleep 5
done; done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
n=$(ls $OUT/*/*_addition.json 2>/dev/null | wc -l); echo "jsons=$n/20"
[ "$n" -eq 20 ] && touch "$OUT/.done"
echo "pilot finished $(date)"
