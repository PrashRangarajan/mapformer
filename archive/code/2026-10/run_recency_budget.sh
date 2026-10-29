#!/usr/bin/env bash
# Rule 5: is the index arms' 0.234 a capability limit or a budget limit?
# Their loss slope over the final 10% is -0.005/epoch, i.e. still falling, so
# the 300-epoch number is not licensed as a converged one. 2x budget, 3 seeds.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/recency_budget; mkdir -p "$OUT" "$OUT/logs"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
for s in 0 1 2; do for v in RoPE PlainFlat Signed_r4; do
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] && continue
  while [ "$(busy)" -ge 6 ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s ep600 -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" \
    --k-max 64 --T 1024 --eval-T 1024 2048 --epochs 600 --n-batches 48 \
    --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 \
    --n-heads 2 --fast-attn --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
    > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 8
done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
touch "$REPO/.recency_budget_done"; echo "done $(date)"
