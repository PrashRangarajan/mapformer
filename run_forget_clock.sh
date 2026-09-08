#!/usr/bin/env bash
# Chained: waits for the compositional batch, then runs the forget-gate clock test.
# See FORGET_CLOCK_PREREG.md.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
# wait for the compositional batch to clear (marker, or nothing left running)
while [ ! -f "$REPO/.comp_headroom_done" ] && [ "$(busy)" -gt 0 ]; do sleep 120; done
echo "compositional clear at $(date); starting forget-clock batch"

OUT=$REPO/runs/forget_clock; mkdir -p "$OUT" "$OUT/logs"
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
slot(){ while [ "$(busy)" -ge 6 ]; do sleep 20; done; }

for s in 0 1 2 3 4 5 6 7; do for v in Vanilla Forget Forget_Frozen; do
  O="$OUT/torus_${v}_s${s}"; [ -f "$O/${v}.pt" ] && continue
  mkdir -p "$O"; slot; g=$(pick); echo "torus $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$O" > "$OUT/logs/torus_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

for s in 0 1 2 3 4 5 6 7; do for v in Vanilla Forget Forget_Frozen; do
  O="$OUT/recency_${v}_s${s}"; [ -f "$O/${v}_recency.json" ] && continue
  mkdir -p "$O"; slot; g=$(pick); echo "recency $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" \
    --k-max 64 --T 1024 --eval-T 1024 2048 4096 --epochs 300 --n-batches 48 \
    --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 \
    --n-heads 2 --device "cuda:$g" --output-dir "$O" \
    > "$OUT/logs/recency_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

while [ "$(busy)" -gt 0 ]; do sleep 30; done
miss=0
for s in 0 1 2 3 4 5 6 7; do for v in Vanilla Forget Forget_Frozen; do
  [ -f "$OUT/torus_${v}_s${s}/${v}.pt" ] || { echo "MISSING torus $v s$s"; miss=$((miss+1)); }
  [ -f "$OUT/recency_${v}_s${s}/${v}_recency.json" ] || { echo "MISSING recency $v s$s"; miss=$((miss+1)); }
done; done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.forget_clock_done"
echo "finished $(date)"
