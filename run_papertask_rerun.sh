#!/usr/bin/env bash
# PAPERTASK_PREREG.md: does EM's extended-length advantage survive a CONVERGED recipe?
# 3 arms x 8 seeds, one batch, cosine + 50 epochs (the original was 16 on LinearLR), and the
# training logs are KEPT -- the previous batch's were deleted, which is why rule 9 could not be
# applied to the one EM-beats-WM cell on a map task.
#
# Waits for the pair-origin batch so the two do not contend for GPUs.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/paper_task_rerun
mkdir -p "$OUT/logs"
MAXPG=3                       # batch 128 x 128 steps is heavier than the recency runs
VARS="Vanilla VanillaEM_P0 MapPoPE-Flat"

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
recency_busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }

echo "$(date +%H:%M) waiting for the pair-origin batch to finish"
until [ -f "$REPO/runs/pairorigin/.done" ] || [ "$(recency_busy)" -eq 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs free; starting"

for s in $(seq 0 7); do
  for v in $VARS; do
    [ -f "$OUT/${v}_s${s}/${v}.pt" ] && { echo "skip $v s$s"; continue; }
    while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
    g=$(pick); echo "launch $v s$s -> cuda:$g"
    setsid nohup python3 -u -m mapformer.train_variant \
      --variant "$v" --seed "$s" --epochs 50 --schedule cosine \
      --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
      > "$OUT/logs/train_${v}_s${s}.log" 2>&1 &
    sleep 8
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done

missing=0
for v in $VARS; do for s in $(seq 0 7); do
  [ -f "$OUT/${v}_s${s}/${v}.pt" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -gt 0 ] && { echo "not evaluating an incomplete batch"; exit 1; }

echo "=== eval: the paper's OOD protocol plus the length extension ==="
python3 -u -m mapformer.eval_paper_ood --runs-dir "$OUT" \
  --variants $VARS --seeds 0 1 2 3 4 5 6 7 \
  --extended --n-batches 8 --batch-size 32 --device cuda:1 \
  --out "$REPO/PAPER_OOD_RERUN.md" > "$OUT/eval_ood.log" 2>&1
echo "eval exit $?"
# The marker must certify the ARTIFACT, not the absence of a crash: this batch touched .done
# after an eval that OOM'd, and the false marker released the next batch early.
if [ -s "$REPO/PAPER_OOD_RERUN.json" ]; then touch "$OUT/.done"; else
  echo "eval produced no json -- NOT touching .done"; exit 1; fi
echo "batch finished $(date)"
