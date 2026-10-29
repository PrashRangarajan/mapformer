#!/usr/bin/env bash
# Multi-seed 2x2 on the hierarchical goal task (confirm the scan's synergy
# signal). 4 variants x seeds {0,1,2}, train T_explore=64, eval {64,128,192,256}.
# Question: is MapWM-Hier reliably best OOD, and does hierarchy help MapFormer
# while hurting plain? Resumable; writes HIERGOAL_MULTISEED.md.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOG="$REPO/hiergoal_multiseed.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
OUTROOT="$REPO/runs/hiergoal_multiseed"
VARIANTS=(Vanilla Hourglass_k2 PlainFlat PlainHourglass)

JOBS=()
for s in 0 1 2; do for v in "${VARIANTS[@]}"; do JOBS+=("$s:$v"); done; done

train_one () {
  local gpu="$1" job="$2" s="${2%%:*}" v="${2##*:}" out
  out="$OUTROOT/seed$s"
  if [ -f "$out/${v}_hiergoal.pt" ]; then echo "$(date +%H:%M) skip $job" >> "$LOG"; return; fi
  echo "$(date +%H:%M) [gpu$gpu] $job" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$v" --seed "$s" \
    --epochs 25 --n-batches 64 --T-explore 64 --T-navigate 64 \
    --eval-explore 64 128 192 256 --n-layers 3 --device "cuda:$gpu" \
    --output-dir "$out" >> "$LOG" 2>&1
}

( for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 0 ] && train_one 0 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) &
P0=$!
( for i in "${!JOBS[@]}"; do [ $((i % 2)) -eq 1 ] && train_one 1 "${JOBS[$i]}"; done
  echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) &
P1=$!
wait $P0 $P1
echo "$(date +%H:%M) TRAINING DONE -> aggregating" >> "$LOG"
python3 -u -m mapformer.agg_hiergoal --runs-dir "$OUTROOT" --seeds 0 1 2 \
  --variants "${VARIANTS[@]}" --lengths 64 128 192 256 \
  --out "$REPO/HIERGOAL_MULTISEED.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) AGGREGATE DONE" >> "$LOG"
touch "$REPO/.hiergoal_ms_done"