#!/usr/bin/env bash
# FIXED-MAP learnability + flip diagnostic (seed 0). The fresh-map task memorised
# its 3k-traj buffer (train loss 0.2, held-out NLL 6.2 = confidently wrong). The
# fixed-map task tests PATH INTEGRATION on a known map (novel walk each episode),
# which needs no in-context generalisation and should learn from a small buffer.
# Seed-0 mini-factorial: {Vanilla=path-int, RoPE=index} x {raw, allo}. Answers
# (a) is it learnable now, (b) does the position effect flip raw->allo, before we
# spend the full 3-seed x 4-variant factorial. Pre-builds the 2 fixed-map buffers
# (raw, allo) then trains 4 arms, <=2 per GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_fixed_diag"; mkdir -p "$R"
LOG="$REPO/mw_fixed_diag.log"; echo "fixed-diag start $(date)" > "$LOG"
G=8; T=512; NBUF=3000

echo "$(date +%H:%M) building 2 fixed-map buffers (raw, allo)" >> "$LOG"
for AF in False True; do
  python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B
e=W(grid_size=$G, seed=0, allocentric=$AF, fixed_map=True)
B(e, $T, $NBUF, 0)" >> "$LOG" 2>&1 &
done
wait
echo "$(date +%H:%M) buffers ready; training 4 arms" >> "$LOG"

i=0
for ALLO in "" "--allocentric"; do
  for V in Vanilla RoPE; do
    TAG=$([ -n "$ALLO" ] && echo allo || echo raw)
    GPU=$(( i % 2 ))
    echo "$(date +%H:%M) $V $TAG -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_miniworld --variant "$V" --seed 0 $ALLO --fixed-map \
      --grid-size $G --n-steps $T --buffer-size $NBUF --epochs 100 \
      --n-batches 180 --batch-size 24 --d-model 256 --n-layers 4 --n-heads 4 \
      --eval-lengths 512 1024 --device "cuda:$GPU" --output-dir "$R" \
      > "$R/${V}_${TAG}.log" 2>&1 &
    i=$((i+1)); sleep 2
  done
done
wait
echo "$(date +%H:%M) training done" >> "$LOG"
{
  echo "## fixed-map seed-0 mini-factorial (nb_acc @ T=512 / T=1024)"
  for TAG in raw allo; do for V in Vanilla RoPE; do
    python3 -c "
import json; r=json.load(open('$R/${V}_${TAG}.json'))
print(f'${V:8s} ${TAG:4s}: nb512={r[\"512\"][\"nb_acc\"]:.3f} nll={r[\"512\"][\"nb_nll\"]:.2f} | nb1024={r[\"1024\"][\"nb_acc\"]:.3f}')"
  done; done
} >> "$LOG" 2>&1
touch "$REPO/.mw_fixed_diag_done"
echo "$(date) DONE" >> "$LOG"
