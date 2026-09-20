#!/usr/bin/env bash
# RECENCY_T3_PREREG.md: 3 arms x 8 seeds on recency, the run_recency.sh recipe verbatim.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/recency_t3; mkdir -p "$OUT/logs"
KMAX=64; T=1024; EPOCHS=300; MAXPG=2
VARIANTS="MapPoPE-Flat MapPoPE_T3pi01 MapPoPE_T3inert"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in 0 1 2 3 4 5 6 7; do
  for v in $VARIANTS; do
    d="$OUT/${v}_s$s"
    [ -f "$d/$v.json" ] && { echo "skip $v s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick); echo "$(date +%H:%M) launch $v s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_recency \
      --variant "$v" --seed "$s" --k-max $KMAX --T $T --eval-T 1024 2048 \
      --epochs $EPOCHS --n-batches 48 --batch-size 16 \
      --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${v}_s$s.log" 2>&1 &
    sleep 8
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
missing=0
for s in 0 1 2 3 4 5 6 7; do for v in $VARIANTS; do
  [ -f "$OUT/${v}_s$s/$v.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
echo "batch finished $(date)"
