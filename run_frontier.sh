#!/usr/bin/env bash
# DEPTH-VS-LOOP FRONTIER, in whichever venue the algorithmic run selects.
#
# THE QUESTION. Does looping substitute for real depth EQUALLY WELL with and
# without path integration? LOOP_HEADROOM measured the loop buying +0.099 on the
# index arm and +0.414 on the path-integrated one (interaction +0.315), but it has
# no depth baseline for the index row, so the two frontiers cannot be compared.
# Eight arms fix that: {index, path-int} x {1, 2, 3 real layers, loop x4}.
#
# WHY NOT THE SINGLE-ROW VERSION first proposed. Loop-matches-depth-at-fewer-
# parameters is already established in that literature (Yang et al. 2311.12424
# report matching a 12-layer transformer with fewer parameters; ALBERT and MoR make
# the same claim). Re-measuring it on our task would be confirmatory. Whether it
# INTERACTS with path integration is not established anywhere.
#
# IT ALSO REPAIRS THE EXISTING EVIDENCE. The current "loop matches 3 real layers at
# a third of the parameters" rests on a depth arm that scored 0.771 +/- 0.263 with
# one seed at 0.143 -- an optimisation failure. Drop that seed and the other seven
# average 0.861 against the loop's 0.870. The claim is probably right and is
# resting on a broken comparison; converged arms would settle it.
#
# VENUE is chosen by decide_frontier.py, not by hand, on a rule fixed before the
# numbers were seen.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
LOG="$REPO/frontier.log"; echo "frontier queued $(date)" > "$LOG"

echo "$(date +%H:%M) waiting for the algorithmic batch" >> "$LOG"
until [ -f "$REPO/.algorithmic_done" ]; do sleep 60; done
C="train_algo""rithmic"
while [ "$(pgrep -u "$USER" -f "$C" | wc -l)" -gt 0 ]; do sleep 30; done

python3 -u -m mapformer.decide_frontier >> "$LOG" 2>&1
VENUE=$(grep -oP "^VENUE=\K\w+" "$LOG" | tail -1)
echo "$(date +%H:%M) venue = ${VENUE:-unset}" >> "$LOG"

MAXPG=5
gpu_for(){ P="$1"
  while :; do
    N0=$(pgrep -u "$USER" -af "$P" 2>/dev/null | grep -c -- "--device cuda:0" || true)
    N1=$(pgrep -u "$USER" -af "$P" 2>/dev/null | grep -c -- "--device cuda:1" || true)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then echo 0; return; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then echo 1; return; fi
    if [ "$N0" -lt "$MAXPG" ]; then echo 0; return; fi
    if [ "$N1" -lt "$MAXPG" ]; then echo 1; return; fi
    sleep 20
  done
}

if [ "$VENUE" = "algorithmic" ]; then
  R="$REPO/runs/frontier_alg"; mkdir -p "$R"
  A="train_algo""rithmic"
  for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
   for TASK in parity copy; do
    for SPEC in "RoPE 1" "RoPE 2" "RoPE 3" "RoPELooped 1" \
                "Vanilla 1" "Vanilla 2" "Vanilla 3" "Looped 1"; do
      set -- $SPEC; V="$1"; NL="$2"
      OUT="$R/$TASK/${V}_L${NL}_s${SEED}"; mkdir -p "$OUT"
      [ -f "$OUT/${V}_${TASK}.json" ] && continue
      G=$(gpu_for "$A")
      echo "$(date +%H:%M:%S) $TASK $V L$NL s$SEED -> cuda:$G" >> "$LOG"
      python3 -u -m mapformer.train_algorithmic --variant "$V" --task "$TASK" \
        --seed "$SEED" --epochs 300 --n-batches 50 --batch-size 128 \
        --train-length 16 --eval-lengths 16 32 64 128 256 --lr 1e-3 \
        --d-model 128 --n-heads 2 --n-layers "$NL" --schedule cosine \
        --device "cuda:$G" --output-dir "$OUT" \
        > "$R/${TASK}_${V}_L${NL}_s${SEED}.log" 2>&1 &
      sleep 3
    done
   done
  done
  wait
  N=$(find "$R" -name '*.json' | wc -l)
  echo "$(date +%H:%M) $N/256 results" >> "$LOG"
  python3 -u -m mapformer.agg_frontier --runs-dir "$R" --venue algorithmic \
    --out "$REPO/FRONTIER_ALGORITHMIC.md" >> "$LOG" 2>&1
else
  R="$REPO/runs/frontier_mq"; mkdir -p "$R"
  A="train_match_""query"
  for SEED in 0 1 2 3 4 5 6 7; do
    for SPEC in "RoPE 1" "RoPE 2" "RoPE 3" "RoPELooped 1" \
                "Vanilla 1" "Vanilla 2" "Vanilla 3" "Looped 1"; do
      set -- $SPEC; V="$1"; NL="$2"
      OUT="$R/${V}_L${NL}_s${SEED}"; mkdir -p "$OUT"
      [ -f "$OUT/${V}_matchquery.pt" ] && continue
      G=$(gpu_for "$A")
      echo "$(date +%H:%M:%S) MQ $V L$NL s$SEED -> cuda:$G" >> "$LOG"
      python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
        --epochs 600 --lr 1e-3 --n-batches 48 --batch-size 16 --size 128 \
        --n-obs 16 --T-explore 512 --T-query 256 --n-layers "$NL" --d-model 128 \
        --n-heads 2 --schedule cosine --fast-attn --device "cuda:$G" \
        --output-dir "$OUT" > "$R/${V}_L${NL}_s${SEED}.log" 2>&1 &
      sleep 8
    done
  done
  wait
  N=$(find "$R" -name '*_matchquery.pt' | wc -l)
  echo "$(date +%H:%M) $N/64 checkpoints" >> "$LOG"
  python3 -u -m mapformer.agg_frontier --runs-dir "$R" --venue matchquery \
    --out "$REPO/FRONTIER_MATCHQUERY.md" >> "$LOG" 2>&1
fi
touch "$REPO/.frontier_done"; echo "$(date) DONE" >> "$LOG"
