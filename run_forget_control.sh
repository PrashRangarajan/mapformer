#!/usr/bin/env bash
# The control FORGET_GATE.md asks for, and that V4_MULTISEED.md identified and
# never ran: is the +0.086 the GATE, or 259 parameters?
#
# Forget_Frozen has lambda as a BUFFER pinned at zero. Verified before launch:
#   - initialisation identical to Forget on every shared parameter (0.0e+00), so
#     the RNG stream is matched
#   - decay bias identically 0, and the forward pass equals Vanilla with the same
#     weights transplanted (0.0e+00)
#   - lambda absent from parameters(); |grad W_f| = 0, i.e. present and inert
#
# So the three arms are: no gate params / gate params inert / gate params live.
#   Frozen ~ Forget      -> the gain is PARAMETERS and initialisation shift
#   Frozen ~ Vanilla     -> a live lambda is doing the work, even though what it
#                           learns is ~0 and anti-correlated with the gain
#
# ALL THREE ARMS RETRAINED HERE (rule 3), not compared against runs/forget. That
# also makes Vanilla a reproducibility check on the previous batch: identical
# recipe, identical seeds, no --fast-attn, so it should land on the same numbers.
# Match-Query did NOT reproduce across batches (MQ_RANK_2X2 vs LOOP_HEADROOM), so
# this is worth knowing for the torus rather than assumed.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/forget_ctrl; LOG=$REPO/forget_ctrl.log
mkdir -p "$R"; echo "forget control start $(date)" > "$LOG"

MAXPG=3
on_gpu() { ps -u "$USER" -o comm=,args= \
           | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }

for SEED in 0 1 2 3 4 5 6 7; do
  for V in Vanilla Forget Forget_Frozen; do
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
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/24 checkpoints" >> "$LOG"

python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Forget Forget_Frozen --noises 0.0 --seeds 0 1 2 3 4 5 6 7 \
  --lengths 128 512 1024 --n-trials 100 --device cuda:0 \
  --title "Is the forget gate's +0.086 the gate, or 259 parameters?" \
  --out "$REPO/FORGET_CONTROL.md" >> "$LOG" 2>&1

if [ -f "$REPO/FORGET_CONTROL.md" ]; then
  touch "$REPO/.forget_ctrl_done"; echo "$(date) DONE" >> "$LOG"
else
  echo "$(date) EVAL FAILED" >> "$LOG"
fi
