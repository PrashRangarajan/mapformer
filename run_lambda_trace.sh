#!/usr/bin/env bash
# Log lambda over training, to test the transient-training-aid hypothesis.
#
# FORGET_CONTROL.md left an oddity: a TRAINABLE lambda that ends near zero is
# worth +0.081, while the same architecture with lambda PINNED at zero is worth
# nothing (-0.016) -- and the gain is ANTI-correlated with the final lambda
# (r = -0.516, five of eight seeds ending negative). The hypothesis is that the
# gate is a transient aid: lambda rises early, supplies a recency prior that makes
# the short-lag retrievals the first ones learned, then anneals away once the map
# exists. The 60-step trace taken before that batch shows 0 -> +0.034 against
# final values averaging +0.012, which is suggestive and far too short.
#
# PRE-REGISTERED, and this is the sharp one: if the gate works through its
# TRAJECTORY rather than its endpoint, then across seeds
#
#     r(PEAK lambda, gain)  >  0        while   r(final lambda, gain) = -0.516
#
# A rise-then-fall shape with a positive peak correlation supports it. Monotone
# drift to the final value, or a peak correlation that is also negative, does not.
#
# PAIRING IS LEGITIMATE HERE because the torus retrains bit-identically
# (FORGET_CONTROL.md: 0.0000 max per-seed drift across two independent batches),
# so these runs reproduce the stored Forget checkpoints exactly and their
# trajectories pair with the per-seed gains already measured. That check is what
# licenses this design; without it the traces would belong to different runs than
# the gains.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/lamtrace; LOG=$REPO/lamtrace.log
mkdir -p "$R"; echo "lambda trace queued $(date)" > "$LOG"

busy() { ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
while [ ! -f "$REPO/.popewrap_done" ] || [ "$(busy)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs clear, starting" >> "$LOG"

MAXPG=3
on_gpu() { ps -u "$USER" -o comm=,args= \
           | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }

for SEED in 0 1 2 3 4 5 6 7; do
  OUT="$R/p0/Forget_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/Forget.pt" ] && continue
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
    sleep 20
  done
  echo "$(date +%H:%M:%S) Forget s$SEED -> cuda:$G" >> "$LOG"
  MF_LAM_LOG="$R/lam_s${SEED}.txt" \
  python3 -u -m mapformer.train_variant --variant Forget --seed "$SEED" \
    --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
    > "$R/Forget_s${SEED}.log" 2>&1 &
  sleep 6
done
wait
echo "$(date +%H:%M) $(ls "$R"/lam_s*.txt 2>/dev/null | wc -l)/8 traces" >> "$LOG"
python3 -u -m mapformer.agg_lambda_trace --trace-dir "$R" \
  --gains-json "$REPO/FORGET_CONTROL.json" \
  --out "$REPO/LAMBDA_TRACE.md" >> "$LOG" 2>&1
[ -f "$REPO/LAMBDA_TRACE.md" ] && { touch "$REPO/.lamtrace_done"; echo "$(date) DONE" >> "$LOG"; } \
  || echo "$(date) AGGREGATION FAILED" >> "$LOG"
