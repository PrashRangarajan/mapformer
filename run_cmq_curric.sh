#!/usr/bin/env bash
# Curriculum probe: does ramping the blind horizon T_query (short->long) raise
# the good-basin hit-rate vs warmup-only? Two variants that matter x 3 seeds.
# 3 concurrent per GPU (Vanilla on GPU0, CoarseIdx on GPU1) -> no OOM.
# Compares against the warmup-only stab runs (runs/cmq_stab, same seeds).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOGD="$REPO/runs/cmq_curric"; mkdir -p "$LOGD"
LOG="$REPO/cmq_curric.log"; echo "curric start $(date)" > "$LOG"
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

for S in 0 1 2; do run Vanilla "$S" 0; done            # 3 on GPU0
for S in 0 1 2; do run Hourglass_CoarseIdx "$S" 1; done # 3 on GPU1
wait

echo "$(date) done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" \
    --variants Vanilla Hourglass_CoarseIdx \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_CURRIC.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_curric_done"
echo "$(date) DONE" >> "$LOG"
