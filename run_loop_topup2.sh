#!/usr/bin/env bash
# SECOND TOP-UP: bring PI_L3 and both index arms to n=8, matching the PI pair.
#
# WHY. At n=8 the loop-on-path-integration effect is +0.414 (sd 0.279, MDE 0.277,
# 7/8 positive) -- detectable. The two claims that lean on it are still at n=3 and
# both are shaky:
#
#   "the loop BEATS three real layers by +0.273".  PI_L3 is 0.836 / 0.812 / 0.143
#   -- mean 0.597, sd 0.394. That third seed is a catastrophic basin failure, and
#   at n=2 I had read the first two as "reproduces the published 0.823". Whether
#   the loop beats depth or merely matches it depends entirely on how often L3
#   fails, which n=3 cannot estimate.
#
#   "the interaction is +0.333".  It is (loop on path-int) minus (loop on index),
#   and the index half is n=3.
#
# Five more seeds on each of the three arms. Same settings as the main batch, so
# seeds pool (the pipeline is bit-reproducible: the aliasing repro control drifted
# 0.000).
#
# This is the third time this session a confident reading off n<=3 dissolved with
# more seeds -- the horizon table's "scale hurts", my "92% of what depth buys" on
# the torus, and "L3 reproduces the published number". Standing rule 6 exists for
# exactly this and I keep re-learning it.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/loop_headroom"
LOG="$REPO/loop_topup2.log"; echo "top-up 2 start $(date)" > "$LOG"
EP=300; NB=48; BS=16; SZ=128; NOBS=16; TE=512; TQ=256; DM=128; NH=2
A="train_match_""query"
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done
MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"; LBL="$3"; NL="$4"
  OUT="$R/$LBL/s$SEED"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $LBL s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 45
  done
  echo "$(date +%H:%M) $LBL (L$NL) s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --size $SZ --n-obs $NOBS \
    --T-explore $TE --T-query $TQ --eval-query 256 512 --n-layers "$NL" \
    --d-model $DM --n-heads $NH --schedule cosine --fast-attn \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${LBL}_s${SEED}.log" 2>&1 &
  sleep 15
}
# interleave the three arms so neither GPU collects only the slow 3-layer jobs
for SEED in 3 4 5 6 7; do
  launch Vanilla    "$SEED" PI_L3   3
  launch RoPE       "$SEED" IX_flat 1
  launch RoPELooped "$SEED" IX_loop 1
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l) checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_loop_headroom --runs-dir "$R" \
    --out "$REPO/LOOP_HEADROOM.md" >> "$LOG" 2>&1
touch "$REPO/.loop_topup2_done"; echo "$(date) DONE" >> "$LOG"
