#!/usr/bin/env bash
# Multi-seed the clock scan (seeds 1,2; seed 0 already done) to de-noise the
# PoPE-transfer result, then re-aggregate over n=3.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/clock_ms.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
VS=(Vanilla Hourglass_k2 Hourglass_CoarsePI PoPE MapPoPE_Hier MapPoPE_CoarseIdx PlainFlat)
ck(){ local g=$1 v=$2 s=$3 o="$REPO/runs/clock_scan/seed$3"
  [ -f "$o/${v}_clock.pt" ] && { echo "$(date +%H:%M) skip $v s$s" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_clock --variant "$v" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
JOBS=(); for s in 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run_job(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"; ck "$g" "$v" "$s"; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run_job 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run_job 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating n=3" >> "$LOG"
python3 -u -m mapformer.agg_clock --runs-dir "$REPO/runs/clock_scan" --seeds 0 1 2 \
  --variants "${VS[@]}" --lengths 64 128 192 256 --out "$REPO/CLOCK_SCAN.md" >> "$LOG" 2>&1
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.clock_ms_done"
