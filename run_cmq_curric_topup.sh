#!/usr/bin/env bash
# Complete the curriculum sweep to n=6: rerun the 7 that OOM'd (eval-phase spike
# pushed CAP=5 over 24 GB). Run at 3/GPU via waves + expandable_segments to
# absorb the eval spike. Same curriculum config as the full sweep -> consistent.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOGD="$REPO/runs/cmq_curric"; LOG="$REPO/cmq_curric_topup.log"
echo "curric-topup start $(date)" > "$LOG"
TE=512; TQ=256; EP=200; NB=48; BS=16; WU=0.05; CF=0.5; TQ0=16

run () {  # $1=variant $2=seed $3=gpu
  local V="$1" S="$2" G="$3" OUT="$LOGD/s$2"
  [ -f "$OUT/${V}_cmq.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  echo "$(date +%H:%M) run $V s$S -> cuda:$G" >> "$LOG"
  python3 -u -m mapformer.train_compositional_match_query --variant "$V" --seed "$S" \
      --T-explore $TE --T-query $TQ --tq-start $TQ0 --curriculum-frac $CF \
      --warmup-frac $WU --epochs $EP --n-batches $NB --batch-size $BS \
      --eval-query 256 512 --device "cuda:$G" --output-dir "$OUT" \
      > "$LOGD/${V}_s${S}.log" 2>&1 &
}

# wave 1: 3 per GPU
run PlainFlat 2 0; run Vanilla 3 0; run PlainFlat 3 0
run Hourglass_k2 4 1; run Hourglass_CoarseIdx 4 1; run PlainFlat 4 1
wait
# wave 2: the 7th
run PlainHourglass 4 0
wait

echo "$(date) done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" \
    --variants Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_CURRIC.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_curric_topup_done"
echo "$(date) DONE" >> "$LOG"
