#!/usr/bin/env bash
# PAPER2X2_PREREG.md: the paper-task 2x2 under the converged recipe. Waits for MONOTONE.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/paper2x2
mkdir -p "$OUT/p0"
MAXPG=4
ARMS="RoPE PoPE-Flat Vanilla MapPoPE-Flat Vanilla_r4 MapPoPE_r4"
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_/ && index($0,g)' | wc -l; }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
monotone_running(){ ps -u "$USER" -o comm=,args= | awk '$1=="bash" && $2 !~ /^-/ && /run_monotone\.sh$/' | grep -q .; }

echo "$(date +%H:%M) waiting for MONOTONE"
while monotone_running; do sleep 60; done
echo "$(date +%H:%M) starting"

for s in $(seq 0 7); do for v in $ARMS; do
  d="$OUT/p0/${v}_s${s}"
  [ -f "$d/${v}.pt" ] && { echo "skip $v s$s"; continue; }
  while :; do a=$(on_gpu 0); b=$(on_gpu 1)
    if [ "$a" -le "$b" ] && [ "$a" -lt $MAXPG ]; then g=0; break; fi
    if [ "$b" -lt $MAXPG ]; then g=1; break; fi
    sleep 20; done
  mkdir -p "$d"; echo "$(date +%H:%M) $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$d" > "$OUT/${v}_s${s}.log" 2>&1 &
  sleep 8
done; done
while [ "$(busy)" -gt 0 ]; do sleep 30; done

missing=0
for v in $ARMS; do for s in $(seq 0 7); do
  [ -f "$OUT/p0/${v}_s${s}/${v}.pt" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }; done; done
echo "missing=$missing"
[ "$missing" -gt 0 ] && { echo "not evaluating an incomplete batch"; exit 1; }

python3 -u -m mapformer.eval_noise_refine --runs-dir "$OUT" --variants $ARMS --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 128 512 1024 --n-trials 100 --device cuda:0 \
  --out "$REPO/_PAPER2X2_RAW.md" --title "PAPER2X2 raw evaluation" > "$OUT/eval.log" 2>&1
echo "eval exit $?"
[ -s "$REPO/_PAPER2X2_RAW.json" ] || { echo "no eval json -- stopping"; exit 1; }
python3 -u -m mapformer.analyze_paper2x2 > "$OUT/analyze.log" 2>&1
echo "analysis exit $?"
if [ -s "$REPO/PAPER2X2_RESULTS.md" ]; then touch "$OUT/.done"; else echo "no results file -- NOT touching .done"; exit 1; fi
echo "batch finished $(date)"
