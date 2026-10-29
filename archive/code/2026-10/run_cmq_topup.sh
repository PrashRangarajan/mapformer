#!/usr/bin/env bash
# Top-up the two runs that OOM'd from over-packing GPU0 (PlainFlat s2,
# PlainHourglass s2). Wait until a GPU has headroom, run them SEQUENTIALLY there
# (one at a time -> no OOM), then re-aggregate. Resumable: skips if a checkpoint
# already exists.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOGD="$REPO/runs/cmq_sweep"
LOG="$REPO/cmq_topup.log"
echo "topup start $(date)" > "$LOG"

free_mb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
pick_gpu () {   # echo a gpu index with > 6000 MiB free, else nothing
  for g in 0 1; do [ "$(free_mb $g)" -gt 6000 ] && { echo "$g"; return; }; done
}

run_one () {   # $1 variant
  local V="$1"
  [ -f "$LOGD/s2/${V}_cmq.pt" ] && { echo "skip $V s2 (exists)" >> "$LOG"; return; }
  local G=""
  until G=$(pick_gpu); [ -n "$G" ]; do
    echo "$(date +%H:%M) waiting for GPU headroom (g0=$(free_mb 0) g1=$(free_mb 1))" >> "$LOG"
    sleep 60
  done
  echo "$(date +%H:%M) run $V s2 on cuda:$G" >> "$LOG"
  python3 -u -m mapformer.train_compositional_match_query \
      --variant "$V" --seed 2 --T-explore 512 --T-query 256 --epochs 200 \
      --n-batches 48 --batch-size 16 --eval-query 256 512 \
      --device "cuda:$G" --output-dir "$LOGD/s2" >> "$LOGD/${V}_s2.log" 2>&1
}

run_one PlainFlat
run_one PlainHourglass

python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" \
    --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_RESULTS.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_topup_done"
echo "$(date) topup DONE" >> "$LOG"
