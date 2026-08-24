#!/usr/bin/env bash
# Complete the stabilised sweep to n=6 by rerunning the 7 runs that OOM'd.
# Uses the SAME config/code as the original 23 (warmup 0.05, TE512/TQ256, 200ep)
# so the batch stays consistent. Per-GPU cap is guaranteed by wave structure
# (<=3 per GPU -> ~9 GB, no OOM). Resumable. Re-aggregates at the end.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOGD="$REPO/runs/cmq_stab"; LOG="$REPO/cmq_stab_topup.log"
echo "topup start $(date)" > "$LOG"
WU=0.05; TE=512; TQ=256; EP=200; NB=48; BS=16

run () {  # $1=variant $2=seed $3=gpu  (backgrounded)
  local V="$1" S="$2" G="$3" OUT="$LOGD/s$2"
  if [ -f "$OUT/${V}_cmq.pt" ]; then echo "skip $V s$S (exists)" >> "$LOG"; return; fi
  echo "$(date +%H:%M) run $V s$S -> cuda:$G" >> "$LOG"
  python3 -u -m mapformer.train_compositional_match_query --variant "$V" --seed "$S" \
      --T-explore $TE --T-query $TQ --epochs $EP --n-batches $NB --batch-size $BS \
      --warmup-frac $WU --eval-query 256 512 --device "cuda:$G" --output-dir "$OUT" \
      > "$LOGD/${V}_s${S}.log" 2>&1 &
}

# wave 1: 3 on each GPU
run PlainHourglass 2 0; run Hourglass_k2 3 0; run PlainFlat 3 0
run Hourglass_k2 4 1; run PlainHourglass 4 1; run Hourglass_CoarseIdx 5 1
wait
# wave 2: the 7th
run PlainHourglass 5 0
wait

echo "$(date) reruns done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" \
    --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_STAB.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_stab_topup_done"
echo "$(date) DONE" >> "$LOG"
