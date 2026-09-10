#!/usr/bin/env bash
# Phase degrees of freedom at MATCHED initial coherence. Pre-reg: DOF_PREREG.md.
# 4 arms x 8 seeds x 2 tasks; each TASK is one batch (rule 3), including fresh
# VanillaEM_P0_r4 / VanillaEM_r4 rather than reading runs/recency_em.
# RECENCY FIRST -- it carries the discriminator (D1) and is ~3x faster per run.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr

ARMS=(EMDoF_alignfree EMDoF_alignlock VanillaEM_P0_r4 VanillaEM_r4)
SEEDS=(0 1 2 3 4 5 6 7)
MAXPG=3

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_(variant|recency)/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_(variant|recency)/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
slot(){ while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done; }

R_OUT=$REPO/runs/dof/recency; mkdir -p "$R_OUT/logs"
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
touch "$REPO/runs/dof/.recency_done"; echo "recency half done $(date)"

T_OUT=$REPO/runs/dof/torus; mkdir -p "$T_OUT/logs"
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
while [ "$(busy)" -gt 0 ]; do sleep 30; done

# The evaluator expects runs_dir/<noise-tag>/<v>_s<s>/<v>.pt -- same layout trap
# that silently returned all-dashes for N5 and for eval_vocab_sweep before it.
mkdir -p "$REPO/runs/dof/torus_eval_stage"
ln -sfn "$T_OUT" "$REPO/runs/dof/torus_eval_stage/p0"

missing=0
for s in "${SEEDS[@]}"; do for v in "${ARMS[@]}"; do
  [ -f "$R_OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING recency $v s$s"; missing=$((missing+1)); }
  [ -f "$T_OUT/${v}_s${s}/${v}.pt" ] || { echo "MISSING torus $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$REPO/runs/dof/.done"
echo "batch finished $(date)"
