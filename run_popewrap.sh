#!/usr/bin/env bash
# Does PoPE's gain scale with how much the phase WRAPS?
#
# PREDICTION (mapformer_math.tex sec 3.1). PoPE's two changes are one forced
# trade: a rotation acts on a 2-plane and has one angle, so d_head coordinates
# admit exactly d_head/2 rotary frequencies. PoPE escapes that ceiling only by not
# rotating -- it builds mu e^{i phi} elementwise -- paying the per-element phase
# for a per-element frequency at the same d_head content budget.
#
# So the doubled count should pay exactly when the spectrum is stretched thin.
# omega is initialised over [2pi/N, 2pi] with a FIXED 32 blocks, so it spans
# log2(N) octaves and the log-spacing coarsens as N grows:
#
#     grid   16   32   64   128
#     oct     4    5    6     7      (measured, not assumed)
#
# PRE-REGISTERED: the MapPoPE - Vanilla gain rises monotonically with grid size,
# and rises with evaluation length at fixed grid (path length is the other
# wrapping knob). FALSIFIED if flat across grids, or decreasing.
#
# GATED FIRST. Grid 8 is EXCLUDED: 82 scored positions per trajectory against ~29
# elsewhere, a different majority-class rate (0.472 vs ~0.52), and known solvable
# without position at all. Revisit rate plateaus at 0.225 from grid 32 up, so
# label counts are comparable across 32/64/128. A null at small grid is only
# informative if both arms are off ceiling; absolute means are reported beside
# every contrast so that is checkable rather than assumed.
#
# Parameter-matched at every grid: 204,373 vs 204,693, +320 throughout.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/popewrap; LOG=$REPO/popewrap.log
mkdir -p "$R"; echo "popewrap start $(date)" > "$LOG"

busy() { ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
while [ "$(busy)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs clear, starting" >> "$LOG"

GRIDS="16 32 64 128"; MAXPG=3
on_gpu() { ps -u "$USER" -o comm=,args= \
           | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_variant/ && index($0,d)' | wc -l; }

for SEED in 0 1 2 3 4 5 6 7; do
  for N in $GRIDS; do
    for V in Vanilla MapPoPE; do
      OUT="$R/g${N}/p0/${V}_s${SEED}"; mkdir -p "$OUT"
      [ -f "$OUT/${V}.pt" ] && continue
      while :; do
        N0=$(on_gpu 0); N1=$(on_gpu 1)
        if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
        if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
        if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
        sleep 20
      done
      echo "$(date +%H:%M:%S) grid$N $V s$SEED -> cuda:$G" >> "$LOG"
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --grid-size "$N" --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 \
        --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        --schedule cosine --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
        > "$R/g${N}_${V}_s${SEED}.log" 2>&1 &
      sleep 6
    done
  done
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/64 checkpoints" >> "$LOG"

for N in $GRIDS; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/g${N}" \
    --variants Vanilla MapPoPE --noises 0.0 --seeds 0 1 2 3 4 5 6 7 \
    --lengths 128 512 1024 --n-trials 100 --grid-size "$N" --device cuda:0 \
    --title "grid $N" --out "$R/g${N}/acc.md" >> "$LOG" 2>&1
done
python3 -u -m mapformer.agg_popewrap --runs-dir "$R" --grids $GRIDS \
  --arms Vanilla MapPoPE --out "$REPO/POPE_WRAPPING.md" >> "$LOG" 2>&1

if [ -f "$REPO/POPE_WRAPPING.md" ]; then
  touch "$REPO/.popewrap_done"; echo "$(date) DONE" >> "$LOG"
else
  echo "$(date) AGGREGATION FAILED" >> "$LOG"
fi
