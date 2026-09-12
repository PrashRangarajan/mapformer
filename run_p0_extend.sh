#!/usr/bin/env bash
# Extend VanillaEM_P0_r4 to seeds 24-47 so PAIRSPLIT's C2 (the pathway term) can be read at
# n=48 like C1. Same recipe and same directory as the existing P0 arm; a bitwise determinism
# re-check of s0 licenses mixing the new seeds with the stored ones.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/dof/recency
REP=$REPO/runs/p0_extend_repro
mkdir -p "$OUT/logs" "$REP/logs"
MAXPG=4
mine(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /VanillaEM_P0_r4/' | wc -l; }
others(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && !/VanillaEM_P0_r4/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /VanillaEM_P0_r4/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local s=$1 dir=$2
  [ -f "$dir/VanillaEM_P0_r4_s${s}/VanillaEM_P0_r4_recency.json" ] && { echo "skip s$s"; return; }
  while [ "$(mine)" -ge $((2*MAXPG)) ] || [ "$(others)" -gt 0 ]; do sleep 30; done
  g=$(pick); echo "launch P0 s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant VanillaEM_P0_r4 --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$dir/VanillaEM_P0_r4_s${s}" \
    > "$dir/logs/VanillaEM_P0_r4_s${s}.log" 2>&1 &
  sleep 6; }
echo "$(date +%H:%M) waiting for PAIRSPLIT to clear"
until [ "$(others)" -eq 0 ]; do sleep 60; done
go 0 "$REP"                                  # determinism re-check
for s in $(seq 24 47); do go "$s" "$OUT"; done
while [ "$(mine)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$REP/DETERMINISM.txt" 2>&1
from mapformer.ckpt_guard import compare_checkpoints
print(compare_checkpoints("runs/p0_extend_repro/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt",
                          "runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt").report())
PY
missing=0
for s in $(seq 24 47); do [ -f "$OUT/VanillaEM_P0_r4_s${s}/VanillaEM_P0_r4_recency.json" ] || { echo "MISSING s$s"; missing=$((missing+1)); }; done
echo "missing=$missing"; cat "$REP/DETERMINISM.txt"
[ "$missing" -eq 0 ] && touch "$REPO/runs/pairsplit/.p0_extend_done"
echo "batch finished $(date)"
