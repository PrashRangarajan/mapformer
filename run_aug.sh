#!/usr/bin/env bash
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/jsb_aug; mkdir -p "$OUT/logs"; MAXPG=2
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_jsb/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_jsb/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in 0 1 2 3 4; do for A in PoPE MapPoPE; do
  n="$A"; [ "$A" = MapPoPE ] && n="MapPoPE_r2"; n="${n}_aug3"
  d="$OUT/${n}_s$s"; [ -f "$d/$n.json" ] && { echo "skip $n s$s"; continue; }
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 15; done
  g=$(pick); echo "$(date +%H:%M) $n s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_jsb --arch "$A" --augment 3 --seed "$s" \
    --eval-every 250 --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
  sleep 5
done; done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
echo "batch finished $(date)"
