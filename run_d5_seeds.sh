#!/usr/bin/env bash
# D5 at n=24: extend runs/dof/recency with seeds 8-23. Pre-reg: D5_PREREG.md.
# All four arms extended so the decomposition stays on one n.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
ARMS=(EMDoF_alignfree EMDoF_alignlock VanillaEM_P0_r4 VanillaEM_r4)
SEEDS=(8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
MAXPG=3
OUT=$REPO/runs/dof/recency; mkdir -p "$OUT/logs"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$OUT/${v}_s${s}/${v}_recency.pt" ] && { echo "skip $v s$s"; continue; }
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
    > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 8
done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
missing=0
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$REPO/runs/dof/.d5_done"
echo "batch finished $(date)"
