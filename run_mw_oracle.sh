#!/usr/bin/env bash
# ORACLE experiment: the decisive causal test of the reconstruction-fidelity
# explanation for the fresh-map flip failure. Same fresh-map continuous-3D env,
# same in-context demand, same models/budget -- only the ACTION ENCODING fidelity
# changes: {allo = 24-bin direction (R²=0.55, the negative control), oracle =
# exact cell transition (R²->1, clamp rate 0)}. Prediction: path-int FLIPS
# POSITIVE under oracle (24-bin control stays negative in the SAME batch) ->
# confirms the flip failure was quantization/magnitude-variance, not mechanism.
# {Vanilla,MapPoPE}=path-int × {RoPE,PoPE}=index × {allo,oracle} × 3 seeds = 24.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/miniworld_oracle"; mkdir -p "$R"
LOG="$REPO/miniworld_oracle.log"; echo "oracle exp start $(date)" > "$LOG"
VARS="Vanilla MapPoPE-Flat RoPE PoPE-Flat"
G=8; T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

# ---- Phase 0: fresh-map n-gram gate on the ORACLE stream (allo already gated) --
echo "$(date +%H:%M) Phase 0: n-gram gate on oracle stream" >> "$LOG"
python3 -u -m mapformer.validate_miniworld --grids 8 --T $T --n-episodes 40 --oracle \
  --out "$REPO/MINIWORLD_ORACLE_GATES.md" >> "$LOG" 2>&1
if grep -qE "\| G4 action n-gram \| FAIL" "$REPO/MINIWORLD_ORACLE_GATES.md"; then
  echo "ABORT: oracle n-gram gate FAILED" >> "$LOG"; exit 1
fi
echo "$(date +%H:%M) oracle gate PASS" >> "$LOG"

# ---- Phase 1: pre-build buffers (allo + oracle) serially, no shared-key race ---
echo "$(date +%H:%M) Phase 1: pre-building allo+oracle train+eval buffers" >> "$LOG"
for SEED in 0 1 2; do
  python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B
B(W(grid_size=$G, seed=$SEED, allocentric=True,  fixed_map=False), $T, $NBUF, $SEED, n_workers=$NW)
B(W(grid_size=$G, seed=$SEED, oracle=True,        fixed_map=False), $T, $NBUF, $SEED, n_workers=$NW)" >> "$LOG" 2>&1
done
for FLAG in "allocentric=True" "oracle=True"; do
  for L in 512 1024; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_eval_buffer as E
E(W(grid_size=$G, seed=10000, $FLAG, fixed_map=False), $L, $ETRIALS, n_workers=$NW)" >> "$LOG" 2>&1
  done
done
echo "$(date +%H:%M) buffers ready; training 24 arms (<=2/GPU)" >> "$LOG"

# ---- Phase 2: 24 arms, strict <=MAXPG per GPU ----
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for SEED in 0 1 2; do
  for ENC in "--allocentric" "--oracle"; do
    for V in $VARS; do
      TAG=$([ "$ENC" = "--oracle" ] && echo oracle || echo allo); OUT="$R/s${SEED}"
      [ -f "$OUT/${V}_${TAG}.pt" ] && { echo "skip ${V}_s${SEED}_${TAG}" >> "$LOG"; continue; }
      while :; do
        PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
        if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
        if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
        sleep 15
      done
      echo "$(date +%H:%M) ${V}_s${SEED}_${TAG} -> cuda:$GPU" >> "$LOG"
      python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" $ENC \
        --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
        --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
        --eval-trials $ETRIALS --eval-lengths 512 1024 --device "cuda:$GPU" \
        --output-dir "$OUT" > "$R/${V}_s${SEED}_${TAG}.log" 2>&1 &
      PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
    done
  done
done
wait
echo "$(date +%H:%M) training exited; verifying 24 json" >> "$LOG"

# ---- Phase 3: verify, aggregate (allo vs oracle), ablation ----
MISS=0
for SEED in 0 1 2; do for ENC in allo oracle; do for V in $VARS; do
  [ -f "$R/s${SEED}/${V}_${ENC}.json" ] || { echo "MISSING ${V}_s${SEED}_${ENC}" >> "$LOG"; MISS=$((MISS+1)); }
done; done; done
echo "missing: $MISS" >> "$LOG"
if [ "$MISS" -eq 0 ]; then
  for L in 512 1024; do
    python3 -u -m mapformer.agg_miniworld --runs-dir "$R" --length $L --loss-thresh 1.5 \
      --out "$REPO/MINIWORLD_ORACLE_RESULTS_T${L}.md" >> "$LOG" 2>&1
  done
  python3 -u -m mapformer.eval_ablation_miniworld --runs-dir "$R" --length 512 \
    --encodings allo oracle --n-workers $NW --device cuda:0 \
    --out "$REPO/MINIWORLD_ORACLE_ABLATION.md" >> "$LOG" 2>&1
  touch "$REPO/.mw_oracle_done"
fi
echo "$(date) DONE (miss=$MISS)" >> "$LOG"
