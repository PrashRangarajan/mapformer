#!/usr/bin/env bash
# MiniWorld continuous-3D factorial: {path-int, index} x {RoPE, PoPE} x
# {raw, allocentric} x 3 seeds = 24 runs. Question: does the position effect
# (path-int - index) flip from <=0 (raw rotation actions) to positive
# (allocentric displacement), as it did on MiniGrid?
# Phase 1 pre-builds the 6 buffers (3 seeds x raw/allo) in PARALLEL so the 24
# training runs LOAD rather than race to build. Phase 2 trains (<=6/GPU).
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/miniworld"; mkdir -p "$R"
LOG="$REPO/miniworld.log"; echo "miniworld start $(date)" > "$LOG"
VARS="Vanilla MapPoPE-Flat RoPE PoPE-Flat"
G=8; T=512; NBUF=3000; EP=40; NB=120; BS=64

# ---- Phase 1: pre-build 6 buffers in parallel ----
echo "$(date +%H:%M) building 6 buffers" >> "$LOG"
for SEED in 0 1 2; do
  for AF in False True; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B
e=W(grid_size=$G, seed=$SEED, allocentric=$AF)
B(e, $T, $NBUF, $SEED)" >> "$LOG" 2>&1 &
  done
done
wait
echo "$(date +%H:%M) buffers ready; training" >> "$LOG"

# ---- Phase 2: 24 runs, <=6 per GPU ----
i=0
for SEED in 0 1 2; do
  for ALLO in "" "--allocentric"; do
    for V in $VARS; do
      TAG=$([ -n "$ALLO" ] && echo allo || echo raw)
      OUT="$R/s${SEED}"
      [ -f "$OUT/${V}_${TAG}.pt" ] && { echo "skip $V s$SEED $TAG" >> "$LOG"; i=$((i+1)); continue; }
      while [ "$(jobs -rp | wc -l)" -ge 12 ]; do sleep 10; done
      GPU=$(( i % 2 ))
      echo "$(date +%H:%M) $V s$SEED $TAG -> cuda:$GPU" >> "$LOG"
      python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" $ALLO \
        --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP \
        --n-batches $NB --batch-size $BS --eval-lengths 512 1024 \
        --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}_${TAG}.log" 2>&1 &
      i=$((i+1)); sleep 2
    done
  done
done
wait
echo "$(date +%H:%M) training done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_miniworld --runs-dir "$R" --length 512 \
  --out "$REPO/MINIWORLD_RESULTS.md" >> "$LOG" 2>&1
touch "$REPO/.miniworld_done"
echo "$(date) DONE" >> "$LOG"
