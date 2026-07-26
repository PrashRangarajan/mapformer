#!/usr/bin/env bash
# Multi-seed compositional comparison (koopman).
#   6 variants x 3 seeds, split across both GPUs, then aggregate mean+-std.
# Variants: the 4 MapFormer Phase-1 variants + the 2 non-MapFormer controls.
# Resumable: skips any (seed,variant) whose checkpoint already exists.
# Produces COMPOSITIONAL_MULTISEED.md / .json. Does NOT commit/push.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."                     # run as `python3 -m mapformer.X` from parent
LOG="$REPO/compositional_multiseed.log"
: > "$LOG"
echo "multiseed start $(date)" >> "$LOG"

SEEDS=(0 1 2)
VARIANTS=(Vanilla VanillaEM Hourglass_k2 HourglassFlat3 PlainHourglass PlainFlat)
OUTROOT="$REPO/runs/comp_multiseed"

JOBS=()
for s in "${SEEDS[@]}"; do for v in "${VARIANTS[@]}"; do JOBS+=("$s:$v"); done; done

train_one () {
  local gpu="$1" job="$2"
  local s="${job%%:*}" v="${job##*:}"
  local out="$OUTROOT/seed$s"
  if [ -f "$out/$v.pt" ]; then
    echo "$(date +%H:%M) [gpu$gpu] skip $job (checkpoint exists)" >> "$LOG"; return
  fi
  echo "$(date +%H:%M) [gpu$gpu] train $job" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$v" --target motif \
    --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 \
    --seed "$s" --device "cuda:$gpu" --output-dir "$out" >> "$LOG" 2>&1
}

# Two GPU chains: even-index jobs -> gpu0, odd-index jobs -> gpu1.
(
  for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 0 ] && train_one 0 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU0 chain DONE" >> "$LOG"
) &
P0=$!
(
  for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 1 ] && train_one 1 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU1 chain DONE" >> "$LOG"
) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) ALL TRAINING DONE -> aggregating" >> "$LOG"

python3 -u -m mapformer.agg_comp_multiseed \
  --runs-dir "$OUTROOT" --seeds "${SEEDS[@]}" \
  --variants "${VARIANTS[@]}" \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/COMPOSITIONAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) AGGREGATE DONE" >> "$LOG"
touch "$REPO/.comp_multiseed_done"
