#!/usr/bin/env bash
# Full curriculum sweep: 5 variants x 6 seeds = 30 runs, T_query curriculum
# (16->256) + LR warmup. Reuses the 6 probe runs already in runs/cmq_curric
# (resumable skip). PROPER per-GPU cap (CAP live jobs per GPU) -- fixes the
# total-only cap that let one GPU overfill and OOM.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOGD="$REPO/runs/cmq_curric"; mkdir -p "$LOGD"
LOG="$REPO/cmq_curric_full.log"; echo "curric-full start $(date)" > "$LOG"
TE=512; TQ=256; EP=200; NB=48; BS=16; WU=0.05; CF=0.5; TQ0=16
CAP=5                                  # max live runs per GPU (~12.5 GB, safe)

VARIANTS=(Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass)
SEEDS=(0 1 2 3 4 5)

declare -a PIDS_0=() PIDS_1=()
live_count () {                        # $1=gpu ; prune finished, echo live count
  local -n arr="PIDS_$1"; local keep=()
  for p in ${arr[@]+"${arr[@]}"}; do kill -0 "$p" 2>/dev/null && keep+=("$p"); done
  arr=(${keep[@]+"${keep[@]}"}); echo "${#arr[@]}"
}
launch () {                            # $1=variant $2=seed ; picks a free GPU slot
  local V="$1" S="$2" OUT="$LOGD/s$2"
  if [ -f "$OUT/${V}_cmq.pt" ]; then echo "skip $V s$S (exists)" >> "$LOG"; return; fi
  local G=""
  while true; do
    for g in 0 1; do [ "$(live_count $g)" -lt $CAP ] && { G=$g; break; }; done
    [ -n "$G" ] && break; sleep 10
  done
  echo "$(date +%H:%M) run $V s$S -> cuda:$G" >> "$LOG"
  python3 -u -m mapformer.train_compositional_match_query --variant "$V" --seed "$S" \
      --T-explore $TE --T-query $TQ --tq-start $TQ0 --curriculum-frac $CF \
      --warmup-frac $WU --epochs $EP --n-batches $NB --batch-size $BS \
      --eval-query 256 512 --device "cuda:$G" --output-dir "$OUT" \
      > "$LOGD/${V}_s${S}.log" 2>&1 &
  local pid=$!
  if [ "$G" = 0 ]; then PIDS_0+=("$pid"); else PIDS_1+=("$pid"); fi
  sleep 2
}

for S in "${SEEDS[@]}"; do for V in "${VARIANTS[@]}"; do launch "$V" "$S"; done; done
wait

echo "$(date) all done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" --variants "${VARIANTS[@]}" \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_CURRIC.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_curric_full_done"
echo "$(date) DONE" >> "$LOG"
