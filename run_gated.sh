#!/usr/bin/env bash
# The GATED SIGNED variant on BOTH tasks. See GATED_PREREG.md.
# Torus recipe copied verbatim from run_sign.sh so the arms are comparable to the
# sign ablation; recency recipe copied from run_recency.sh for the same reason.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
mkdir -p "$REPO/runs/gated_torus" "$REPO/runs/gated_recency" "$REPO/runs/gated_logs"

ARMS="Gated_r4 Gated_r4_frozen Vanilla_r4 Gated_r2"
SEEDS="0 1 2 3 4 5 6 7"
MAXPG=3

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
wait_slot(){ while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done; }

# ---- torus: the recipe of run_sign.sh, verbatim ----
for s in $SEEDS; do for v in $ARMS; do
  O="$REPO/runs/gated_torus/${v}_s${s}"
  [ -f "$O/${v}.pt" ] && continue
  mkdir -p "$O"; wait_slot; g=$(pick)
  echo "torus $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$O" \
    > "$REPO/runs/gated_logs/torus_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

# ---- recency: the recipe of run_recency.sh, verbatim ----
for s in $SEEDS; do for v in $ARMS; do
  O="$REPO/runs/gated_recency/${v}_s${s}"
  [ -f "$O/${v}_recency.pt" ] && continue
  mkdir -p "$O"; wait_slot; g=$(pick)
  echo "recency $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant "$v" --seed "$s" \
    --k-max 64 --T 1024 --eval-T 1024 2048 --epochs 300 --n-batches 48 \
    --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 \
    --n-heads 2 --fast-attn --device "cuda:$g" --output-dir "$O" \
    > "$REPO/runs/gated_logs/recency_${v}_s${s}.log" 2>&1 &
  sleep 6
done; done

while [ "$(busy)" -gt 0 ]; do sleep 30; done

# verify the ARTIFACT, since `wait` returns regardless of child success
miss=0
for s in $SEEDS; do for v in $ARMS; do
  [ -f "$REPO/runs/gated_torus/${v}_s${s}/${v}.pt" ] || { echo "MISSING torus $v s$s"; miss=$((miss+1)); }
  [ -f "$REPO/runs/gated_recency/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING recency $v s$s"; miss=$((miss+1)); }
done; done
echo "missing=$miss"
[ "$miss" -eq 0 ] && touch "$REPO/.gated_done"
echo "gated batch finished $(date)"
