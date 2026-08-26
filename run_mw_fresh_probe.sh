#!/usr/bin/env bash
# FRESH-MAP learnability probe. The fixed-map factorial made MiniWorld learnable
# but removed the path-integration signal (a memorised map needs only coarse
# position, which attention supplies). The FLIP should live in the fresh-map /
# in-context regime (like the torus, +0.461), where the map can't be memorised
# and precise position-tracking is required -- and where raw's non-commutative
# turns should genuinely cripple path-int while allocentric rescues it.
# Blocker was data (fresh-map memorised a 3k buffer). Now: parallel build affords
# a BIG buffer. This probe checks a 24k fresh-map buffer actually GENERALISES
# (held-out on a NEW map) before spending the full 6-buffer factorial.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_fresh_probe"; mkdir -p "$R"
LOG="$REPO/mw_fresh_probe.log"; echo "fresh-probe start $(date)" > "$LOG"
G=8; T=512; NBUF=24000; EP=40; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24

echo "$(date +%H:%M) parallel-building seed-0 RAW fresh-map buffer (24k, $NW workers)" >> "$LOG"
python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B
e=W(grid_size=$G, seed=0, allocentric=False, fixed_map=False)
B(e, $T, $NBUF, 0, n_workers=$NW)" >> "$LOG" 2>&1
echo "$(date +%H:%M) buffer ready; training Vanilla + RoPE (raw, fresh)" >> "$LOG"

for VG in "Vanilla 0" "RoPE 1"; do
  read -r V G_ID <<<"$VG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed 0 \
    --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
    --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
    --eval-lengths 512 1024 --device "cuda:$G_ID" --output-dir "$R" \
    > "$R/${V}_raw.log" 2>&1 &
  sleep 2
done
wait
echo "$(date +%H:%M) probe done" >> "$LOG"
{
  echo "## fresh-map probe (held-out on a NEW map; chance nb_acc=0.0625, NLL=2.77)"
  echo "memorisation ceiling to beat: nb_acc 0.27, NLL 4-6 (fresh-map 3k buffer)"
  for V in Vanilla RoPE; do
    python3 -c "
import json; r=json.load(open('$R/${V}_raw.json'))
print('$V raw: nb_acc512=%.3f nll512=%.2f | nb_acc1024=%.3f nll1024=%.2f' % (
  r['512']['nb_acc'], r['512']['nb_nll'], r['1024']['nb_acc'], r['1024']['nb_nll']))"
  done
} >> "$LOG" 2>&1
touch "$REPO/.mw_fresh_probe_done"
echo "$(date) DONE" >> "$LOG"
