#!/usr/bin/env bash
set -u
REPO=/home/prashr/mapformer; OUT=$REPO/runs/counter_batch; mkdir -p "$OUT/logs"; cd /home/prashr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
MAXPG=5
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /counter_batch/' | wc -l; }
for s in 0 1 2 3; do for v in TEMRecency_Query_CounterInstalled WM_Counter EM_Counter VanillaEM_P0_r4 Vanilla_r4; do
  d="$OUT/${v}_s${s}"; [ -f "$d/${v}_recency.json" ] && continue
  while :; do a=$(on_gpu 0); b=$(on_gpu 1)
    if [ "$a" -le "$b" ] && [ "$a" -lt $MAXPG ]; then g=0; break; fi
    if [ "$b" -lt $MAXPG ]; then g=1; break; fi; sleep 20; done
  echo "$(date +%H:%M) $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" --k-max 64 --T 1024 \
    --eval-T 1024 2048 --epochs 300 --n-batches 48 --batch-size 16 --schedule cosine --lr 1e-3 \
    --n-layers 1 --d-model 128 --n-heads 2 --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 6
done; done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
n=$(ls $OUT/*/*_recency.json 2>/dev/null | wc -l); echo "jsons=$n/20"; [ "$n" -eq 20 ] && touch "$OUT/.done"
echo "counter batch finished $(date)"
