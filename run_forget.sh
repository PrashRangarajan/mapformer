#!/usr/bin/env bash
# MapFormer + a forget gate: the empty Re G cell of the unified table.
#
# See model_forget.py for the mechanism and mapformer_math.tex sec 3.3 for where
# the cell comes from. 2x2 {r=2, r=4} x {no gate, gate}, parameter-matched: the
# gate costs exactly +259 on both rows and rank exactly +384 on both, so the
# interaction is clean.
#
# PRE-REGISTERED, WITH A SIGN. The scored event is retrieval of the FIRST visit to
# a cell, median lag 33-47 with a long tail, so a monotone decay penalises exactly
# what is being measured. Prediction: NEUTRAL-TO-NEGATIVE on accuracy. Positive
# would be the surprise. The informative quantity is not the accuracy delta but
# what LAMBDA LEARNS, and a null is interpretable because the start is escapable:
# at lambda = 0 the arm is BIT-IDENTICAL to Vanilla (verified, max|diff| 0.0e+00),
# the gradient there is 5.0e-03 against a 4.8e-04 median parameter, and in a live
# 60-step run lambda leaves zero at step 1 with W_f unlocking by step 2.
# lambda is unconstrained in sign, so anti-recency is reachable and would itself
# be a result.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/forget; LOG=$REPO/forget.log
mkdir -p "$R"; echo "forget start $(date)" > "$LOG"

MAXPG=3
on_gpu() { ps -u "$USER" -o comm=,args= \
           | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }

for SEED in 0 1 2 3 4 5 6 7; do
  for V in Vanilla Vanilla_r4 Forget Forget_r4; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && continue
    while :; do
      N0=$(on_gpu 0); N1=$(on_gpu 1)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      sleep 20
    done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 6
  done
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/32 checkpoints" >> "$LOG"

python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Vanilla_r4 Forget Forget_r4 --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 128 512 1024 --n-trials 100 \
  --device cuda:0 --out "$REPO/FORGET_GATE.md" >> "$LOG" 2>&1
python3 -u -m mapformer.probe_forget --runs-dir "$R/p0" \
  --arms Forget Forget_r4 --out "$REPO/FORGET_GATE.md" >> "$LOG" 2>&1

if [ -f "$REPO/FORGET_GATE.md" ]; then
  touch "$REPO/.forget_done"; echo "$(date) DONE" >> "$LOG"
else
  echo "$(date) EVAL FAILED" >> "$LOG"
fi
