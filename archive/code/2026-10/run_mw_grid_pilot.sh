#!/usr/bin/env bash
# PILOT: does a LARGER grid (24 vs 8) run + learn, fixed-map and fresh-map, and is
# there an early substitutability signal (path-int gaining vs index at large grid)?
# oracle recode (fidelity controlled), seed 0, {Vanilla=path-int, RoPE=index} x
# {fixed, fresh}, reduced budget (40 ep, 8k buffer) for a fast read before the sweep.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_pilot"; mkdir -p "$R"
LOG="$REPO/mw_grid_pilot.log"; echo "grid pilot start $(date)" > "$LOG"
G=24; T=512; NBUF=8000; EP=40; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=64

echo "$(date +%H:%M) building 2 oracle buffers (fixed, fresh) grid=$G" >> "$LOG"
python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B, build_or_load_eval_buffer as E
for fx in (True, False):
    tr=W(grid_size=$G, seed=0, oracle=True, fixed_map=fx)
    B(tr, $T, $NBUF, 0, n_workers=$NW)
    et=W(grid_size=$G, seed=(0 if fx else 10000), oracle=True, fixed_map=fx)
    E(et, $T, $ETRIALS, n_workers=$NW)" >> "$LOG" 2>&1
echo "$(date +%H:%M) buffers ready; training 4 arms" >> "$LOG"

i=0
for MAP in "--fixed-map" ""; do
  for V in Vanilla RoPE; do
    TAG=$([ -n "$MAP" ] && echo fixed || echo fresh)
    GPU=$(( i % 2 ))
    python3 -u -m mapformer.train_miniworld --variant "$V" --seed 0 --oracle $MAP \
      --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
      --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
      --eval-trials $ETRIALS --eval-lengths 512 --device "cuda:$GPU" \
      --output-dir "$R/${TAG}" > "$R/${V}_${TAG}.log" 2>&1 &
    i=$((i+1)); sleep 2
  done
done
wait
echo "$(date +%H:%M) done" >> "$LOG"
{
  echo "## grid=$G oracle pilot (seed0, 40ep) -- chance 0.0625"
  for TAG in fixed fresh; do for V in Vanilla RoPE; do
    python3 -c "
import json,os
p='$R/${TAG}/${V}_oracle.json'
if os.path.exists(p):
  r=json.load(open(p)); print(f'${V:8s} ${TAG:5s}: nb512={r[\"512\"][\"nb_acc\"]:.3f} nll={r[\"512\"][\"nb_nll\"]:.2f}')
else: print(f'${V:8s} ${TAG:5s}: MISSING')"
  done; done
} >> "$LOG" 2>&1
touch "$REPO/.mw_grid_pilot_done"
tail -8 "$LOG"
