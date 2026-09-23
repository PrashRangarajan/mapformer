#!/bin/bash
# DYCK_LADDER_PREREG.md: depth 1-4 at FIXED width (n_heads=2, d=128), 4 arms, 8 seeds.
set -u
exec 9>"/tmp/.mapformer_$(basename "$0" .sh).lock"
flock -n 9 || { echo "another instance already running -- exiting"; exit 0; }
REPO=/home/prashr/mapformer
OUT=$REPO/runs/dyck_ladder
mkdir -p "$OUT"
cd /home/prashr
MAXPG=${MAXPG:-4}
mine(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_dyck/ && /runs\/dyck_ladder/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_dyck/ && index($0,d)' | wc -l; }
lighter(){ [ "$(on_gpu 0)" -le "$(on_gpu 1)" ] && echo 0 || echo 1; }   # balance, never fill-first (rule 13)
for seed in 0 1 2 3 4 5 6 7; do
  for L in 1 2 3 4; do
    for arch in RoPE PoPE MapWM MapPoPE; do
      nm="${arch}-${L}L"; case "$arch" in MapWM|MapPoPE) nm="${nm}_r2";; esac
      [ -f "$OUT/${nm}_s${seed}/${nm}.json" ] && continue
      while [ "$(mine)" -ge "$MAXPG" ]; do sleep 15; done
      g=$(lighter)
      OMP_NUM_THREADS=2 setsid nohup python3 -m mapformer.train_dyck \
        --arch "$arch" --n-layers "$L" --n-heads 2 --rank 2 \
        --seed "$seed" --device "cuda:$g" --output-dir "$OUT/${nm}_s${seed}" \
        >> "$OUT/${nm}_s${seed}.log" 2>&1 &
      sleep 4
    done
  done
  echo "seed $seed dispatched $(date +%H:%M:%S)"
done
while [ "$(mine)" -gt 0 ]; do sleep 20; done
missing=0
for seed in 0 1 2 3 4 5 6 7; do for L in 1 2 3 4; do for arch in RoPE PoPE MapWM MapPoPE; do
  nm="${arch}-${L}L"; case "$arch" in MapWM|MapPoPE) nm="${nm}_r2";; esac
  [ -f "$OUT/${nm}_s${seed}/${nm}.json" ] || { echo "MISSING ${nm}_s${seed}"; missing=$((missing+1)); }
done; done; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "dyck ladder finished $(date)"
