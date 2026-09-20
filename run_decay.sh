#!/usr/bin/env bash
# DECAY_PREREG.md: PoPE_decay and MapPoPE_decay on Bach, 512-token crops, 5 seeds.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/jsb_decay; mkdir -p "$OUT/logs"
MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
rec(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_jsb/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
echo "$(date +%H:%M) waiting for the recency batch"
until [ "$(rec)" -eq 0 ]; do sleep 60; done
for s in 0 1 2 3 4; do
  for A in PoPE_decay MapPoPE_decay; do
    n="$A"; [ "$A" = MapPoPE_decay ] && n="${A}_r2"
    d="$OUT/${n}_s$s"
    [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 15; done
    g=$(pick); echo "$(date +%H:%M) launch $n s$s -> cuda:$g"
    OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch "$A" --seed "$s" \
      --train-len 512 --eval-every 250 --device "cuda:$g" --output-dir "$d" \
      > "$OUT/logs/${n}_s$s.log" 2>&1 &
    sleep 5
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
echo "batch finished $(date)"
