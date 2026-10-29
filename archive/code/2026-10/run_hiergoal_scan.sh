#!/usr/bin/env bash
# Cheap single-seed 2x2 scan on the hierarchical goal task, to check whether the
# variants SEPARATE (before committing to a multi-seed run). Train T_explore=64,
# eval at {64 (train), 128, 192 (OOD explore length)} -- the regime where
# MapFormer's length-robustness should show if this task exercises it.
#   MapWM-Flat=Vanilla  MapWM-Hier=Hourglass_k2  Plain-Flat=PlainFlat  Plain-Hier=PlainHourglass
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOG="$REPO/hiergoal_scan.log"; : > "$LOG"; echo "scan start $(date)" >> "$LOG"
OUT="$REPO/runs/hiergoal_scan"

train_one () {
  local gpu="$1" v="$2"
  echo "$(date +%H:%M) [gpu$gpu] $v" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed 0 \
    --epochs 20 --n-batches 64 --T-explore 64 --T-navigate 64 \
    --eval-explore 64 128 192 --n-layers 3 --device "cuda:$gpu" \
    --output-dir "$OUT" >> "$LOG" 2>&1
}

( train_one 0 Vanilla;      train_one 0 PlainFlat;       echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) &
P0=$!
( train_one 1 Hourglass_k2; train_one 1 PlainHourglass;  echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) SCAN DONE" >> "$LOG"
# compact summary of held-out acc by variant x eval-length
grep -E "^(Vanilla|Hourglass_k2|PlainFlat|PlainHourglass) |HELD-OUT" "$LOG" >> "$LOG.summary" 2>/dev/null || true
touch "$REPO/.hiergoal_scan_done"
