#!/usr/bin/env bash
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/dyck_t2; mkdir -p "$OUT/logs"; MAXPG=3
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_dyck/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_dyck/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in $(seq 0 7); do for A in MapPoPE_abs_T3 MapPoPE_abs_T3inert; do
  n="${A}-1L_r2"; d="$OUT/${n}_s$s"; [ -f "$d/$n.json" ] && continue
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 10; done
  g=$(pick); echo "$(date +%H:%M) $n s$s -> cuda:$g"
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_dyck --arch "$A" --n-layers 1 --n-heads 1 \
    --rank 2 --seed "$s" --device "cuda:$g" --output-dir "$d" > "$OUT/logs/${n}_s$s.log" 2>&1 &
  sleep 3
done; done
while [ "$(busy)" -gt 0 ]; do sleep 15; done
echo "batch finished $(date)"
