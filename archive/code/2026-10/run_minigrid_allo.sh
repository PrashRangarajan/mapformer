#!/usr/bin/env bash
# Allocentric rerun of the MiniGrid DoorKey-16 factorial: identical to
# run_minigrid_2x2x2.sh (same 7 arms, n=3, n_layers=3, 50 ep, obj_color, 25K
# buffer) EXCEPT actions are recorded as realized world-fixed displacement
# instead of turn/forward. Tests whether path integration's MiniGrid deficit
# (position effect -0.017) is a removable input-representation mismatch.
# Both GPUs are ours; cap 4/GPU. Writes MINIGRID_ALLOCENTRIC_2X2X2.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/minigrid_2x2x2_allo"; mkdir -p "$R"
LOG="$REPO/minigrid_allo.log"; echo "allo start $(date)" > "$LOG"
VARS="Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat"

# 1. Pre-build the allocentric cached buffer for each seed (sequential) so the
#    parallel training runs LOAD it rather than 7 racing to build it at once.
echo "$(date +%H:%M) building allocentric buffers" >> "$LOG"
for SEED in 0 1 2; do
  python3 -u -c "from mapformer.minigrid_env import MiniGridWorld_Cached as C; \
e=C(env_name='MiniGrid-DoorKey-16x16-v0',tokenization='obj_color',seed=$SEED,\
buffer_size=25000,allocentric=True); e.generate_batch(2,128)" >> "$LOG" 2>&1
done
echo "$(date +%H:%M) buffers ready; training" >> "$LOG"

# 2. Sweep, 21 runs across both GPUs, cap 8 concurrent (4/GPU).
i=0
for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED" >> "$LOG"; i=$((i+1)); continue; }
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 15; done
    GPU=$(( i % 2 ))
    echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-allocentric --minigrid-cached-buffer 25000 \
      --device "cuda:$GPU" --output-dir "$D" > "$R/train_${V}_s${SEED}.log" 2>&1 &
    i=$((i+1)); sleep 3
  done
done
wait
echo "$(date +%H:%M) training done; evaluating (allocentric)" >> "$LOG"

# 3. Held-out eval with allocentric recoding.
python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir "$R" \
  --variants $VARS --seeds 0 1 2 --lengths 128 512 1024 \
  --allocentric --device cuda:0 \
  --out "$REPO/MINIGRID_ALLOCENTRIC_2X2X2.md" >> "$LOG" 2>&1
touch "$REPO/.minigrid_allo_done"
echo "$(date) DONE" >> "$LOG"
