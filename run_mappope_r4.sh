#!/usr/bin/env bash
# Two questions, one batch each. See MAPPOPE_R4_PREREG.md.
#   Q-A  does r=4 help MapPoPE? Every MapPoPE number in this project is r=2.
#   P4   does the gate help r=2 more than r=4? Needs Vanilla(r=2), which the
#        gated batch omitted -- my design error, this repairs it.
# Torus recipe verbatim from run_sign.sh / run_gated.sh so the arms are comparable.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
mkdir -p "$REPO/runs/mappope_r4" "$REPO/runs/mappope_recency" "$REPO/runs/mappope_logs"

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
wait_slot(){ while [ "$(busy)" -ge 6 ]; do sleep 20; done; }

for s in 0 1 2 3 4 5 6 7; do for v in MapPoPE_r4 MapPoPE-Flat Vanilla Gated_r2; do
  O="$REPO/runs/mappope_r4/${v}_s${s}"; [ -f "$O/${v}.pt" ] && continue
  mkdir -p "$O"; wait_slot; g=$(pick); echo "torus $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$O" \
    > "$REPO/runs/mappope_logs/torus_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

# recency half of P4 -- that is where the r=2 gate signal appeared (+0.020, 7/8)
for s in 0 1 2 3 4 5 6 7; do for v in Vanilla Gated_r2; do
  O="$REPO/runs/mappope_recency/${v}_s${s}"; [ -f "$O/${v}_recency.json" ] && continue
  mkdir -p "$O"; wait_slot; g=$(pick); echo "recency $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" \
    --k-max 64 --T 1024 --eval-T 1024 2048 --epochs 300 --n-batches 48 \
    --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 \
    --n-heads 2 --fast-attn --device "cuda:$g" --output-dir "$O" \
    > "$REPO/runs/mappope_logs/recency_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

while [ "$(busy)" -gt 0 ]; do sleep 30; done
miss=0
for s in 0 1 2 3 4 5 6 7; do
  for v in MapPoPE_r4 MapPoPE-Flat Vanilla Gated_r2; do
    [ -f "$REPO/runs/mappope_r4/${v}_s${s}/${v}.pt" ] || { echo "MISSING torus $v s$s"; miss=$((miss+1)); }; done
  for v in Vanilla Gated_r2; do
    [ -f "$REPO/runs/mappope_recency/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING recency $v s$s"; miss=$((miss+1)); }; done
done
echo "missing=$miss"; [ "$miss" -eq 0 ] && touch "$REPO/.mappope_done"
echo "finished $(date)"
