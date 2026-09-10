#!/usr/bin/env bash
# N5: set the position kernel's coherence rho and see whether its effect INVERTS
# between a map task and a clock task. Pre-registration: N5_PREREG.md.
# 4 arms x 2 tasks x 8 seeds; each TASK is one batch (rule 3).
set -u
REPO=/home/prashr/mapformer
cd /home/prashr

ARMS=(EMPhase_plus_r4 EMPhase_zero_r4 EMPhase_minus_r4 EMPhase_rand_r4)
SEEDS=(0 1 2 3 4 5 6 7)
MAXPG=3

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_(variant|recency)/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_(variant|recency)/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
slot(){ while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done; }

# ---- task 1: torus paper task (map, delta == 0). run_sign.sh recipe.
T_OUT=$REPO/runs/n5_phase/torus; mkdir -p "$T_OUT/logs"
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$T_OUT/${v}_s${s}/${v}.pt" ] && { echo "skip torus $v s$s"; continue; }
  slot; g=$(pick); echo "launch torus $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant \
    --variant "$v" --seed "$s" --epochs 300 --n-batches 98 --batch-size 128 \
    --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --lr 1e-3 --device "cuda:$g" \
    --output-dir "$T_OUT/${v}_s${s}" \
    > "$T_OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 8
done; done

# ---- task 2: recency (clock, delta = k(t)). run_recency_em.sh recipe, no fast-attn.
R_OUT=$REPO/runs/n5_phase/recency; mkdir -p "$R_OUT/logs"
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$R_OUT/${v}_s${s}/${v}_recency.pt" ] && { echo "skip recency $v s$s"; continue; }
  slot; g=$(pick); echo "launch recency $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$R_OUT/${v}_s${s}" \
    > "$R_OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 8
done; done

while [ "$(busy)" -gt 0 ]; do sleep 30; done

missing=0
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$T_OUT/${v}_s${s}/${v}.pt" ] || { echo "MISSING torus $v s$s"; missing=$((missing+1)); }
  [ -f "$R_OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING recency $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$REPO/runs/n5_phase/.done"
echo "batch finished $(date)"
