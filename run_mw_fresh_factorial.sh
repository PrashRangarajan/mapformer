#!/usr/bin/env bash
# FRESH-MAP factorial -- the REAL allocentric-flip test, in the in-context regime
# where path integration is load-bearing (probe: RoPE-index generalises 0.358/
# NLL1.99, Vanilla-path-int RAW collapses to CHANCE 0.093 -- raw turns are not
# cumsum-integrable). Question: does allocentric recoding RESCUE the path-int arm
# (Vanilla raw 0.093 -> allo ~0.35+), flipping the position effect?
#
# All review guardrails baked in:
#  F1 solvable arm present (RoPE already generalises; allo arms = hypothesised winner)
#  F3 fresh-map n-gram gate BOTH encodings, BEFORE training (Phase 0; aborts on leak)
#  F4 NLL-led aggregation
#  F5 100 epochs (match; a 40ep null could be undertraining)
#  code-F3 serial pre-build of every unique buffer BEFORE training (no shared-key race)
#  per-GPU <=2 cap (the OOM bug)
#  eval-buffer cache -> paired held-out set across all arms, no live-eval CPU thrash
# F2 context-destruction ablation is a SEPARATE step (eval_ablation_miniworld.py),
#    run after this completes; it gates any published claim.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/miniworld_fresh"; mkdir -p "$R"
LOG="$REPO/miniworld_fresh.log"; echo "fresh factorial start $(date)" > "$LOG"
VARS="Vanilla MapPoPE-Flat RoPE PoPE-Flat"
G=8; T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

# ---- Phase 0: fresh-map n-gram gate, BOTH encodings (gate BEFORE training) ----
echo "$(date +%H:%M) Phase 0: fresh-map n-gram gate" >> "$LOG"
python3 -u -m mapformer.validate_miniworld --grids 8 --T $T --n-episodes 40 \
  --out "$REPO/MINIWORLD_FRESH_GATES_RAW.md" >> "$LOG" 2>&1
python3 -u -m mapformer.validate_miniworld --grids 8 --T $T --n-episodes 40 --allocentric \
  --out "$REPO/MINIWORLD_FRESH_GATES_ALLO.md" >> "$LOG" 2>&1
if grep -qE "\| G4 action n-gram \| FAIL" "$REPO"/MINIWORLD_FRESH_GATES_*.md; then
  echo "ABORT: fresh-map n-gram gate FAILED -- action stream leaks obs" >> "$LOG"; exit 1
fi
echo "$(date +%H:%M) gate PASS both encodings" >> "$LOG"

# ---- Phase 1: pre-build all unique buffers SERIALLY (no shared-key race) ----
echo "$(date +%H:%M) Phase 1: pre-building 6 train + 4 eval buffers" >> "$LOG"
for SEED in 0 1 2; do
  for AF in False True; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B, build_or_load_eval_buffer as E
tr=W(grid_size=$G, seed=$SEED, allocentric=$AF, fixed_map=False)
B(tr, $T, $NBUF, $SEED, n_workers=$NW)" >> "$LOG" 2>&1
  done
done
# eval buffers: env_test seed=10000, per {allo} x {512,1024}
for AF in False True; do
  for L in 512 1024; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_eval_buffer as E
et=W(grid_size=$G, seed=10000, allocentric=$AF, fixed_map=False)
E(et, $L, $ETRIALS, n_workers=$NW)" >> "$LOG" 2>&1
  done
done
echo "$(date +%H:%M) buffers ready; training 24 arms (<=2/GPU)" >> "$LOG"

# ---- Phase 2: 24 arms, strict <=MAXPG per GPU ----
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for SEED in 0 1 2; do
  for ALLO in "" "--allocentric"; do
    for V in $VARS; do
      TAG=$([ -n "$ALLO" ] && echo allo || echo raw); OUT="$R/s${SEED}"
      [ -f "$OUT/${V}_${TAG}.pt" ] && { echo "skip ${V}_s${SEED}_${TAG}" >> "$LOG"; continue; }
      while :; do
        PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
        if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
        if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
        sleep 15
      done
      echo "$(date +%H:%M) ${V}_s${SEED}_${TAG} -> cuda:$GPU" >> "$LOG"
      python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" $ALLO \
        --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
        --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
        --eval-trials $ETRIALS --eval-lengths 512 1024 --device "cuda:$GPU" \
        --output-dir "$OUT" > "$R/${V}_s${SEED}_${TAG}.log" 2>&1 &
      PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
    done
  done
done
wait
echo "$(date +%H:%M) training exited; verifying all 24 json" >> "$LOG"

# ---- Phase 3: verify (do NOT trust wait), then aggregate ----
MISS=0
for SEED in 0 1 2; do for ALLO in raw allo; do for V in $VARS; do
  [ -f "$R/s${SEED}/${V}_${ALLO}.json" ] || { echo "MISSING ${V}_s${SEED}_${ALLO}" >> "$LOG"; MISS=$((MISS+1)); }
done; done; done
echo "missing: $MISS" >> "$LOG"
if [ "$MISS" -eq 0 ]; then
  # loss-thresh 1.5: the path-int RAW arm is EXPECTED at chance (can't integrate
  # turns) with high loss -- that's the finding, not a stuck arm. Flag only truly
  # catastrophic arms; interpret the table manually (all arms trained in one batch,
  # so the lm200 stale-baseline hazard the guard was built for does not apply).
  python3 -u -m mapformer.agg_miniworld --runs-dir "$R" --length 512 --loss-thresh 1.5 \
    --out "$REPO/MINIWORLD_FRESH_RESULTS.md" >> "$LOG" 2>&1
  python3 -u -m mapformer.agg_miniworld --runs-dir "$R" --length 1024 --loss-thresh 1.5 \
    --out "$REPO/MINIWORLD_FRESH_RESULTS_T1024.md" >> "$LOG" 2>&1
  touch "$REPO/.mw_fresh_done"
fi
echo "$(date) DONE (miss=$MISS)" >> "$LOG"
