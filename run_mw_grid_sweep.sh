#!/usr/bin/env bash
# GRID SWEEP: does path-int gain vs index as the grid grows (revisit distances
# exceed the attention horizon)? The decisive test of the attention-substitutability
# mechanism. Fresh-map, ORACLE recode (fidelity controlled), grids {16,24,32} x
# {Vanilla=path-int, RoPE=index} x 3 seeds = 18 arms, 100 epochs. Grid 8 is reused
# from runs/miniworld_oracle (fresh oracle: Vanilla 0.448, RoPE 0.977, effect -0.53).
# Prediction: Vanilla-RoPE climbs from -0.53 (g8) toward positive as G grows.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"
LOG="$REPO/miniworld_grid_sweep.log"; echo "grid sweep start $(date)" > "$LOG"
GRIDS="16 24 32"; VARS="Vanilla RoPE"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

# ---- Phase 0: n-gram gate on the oracle stream at each grid ----
echo "$(date +%H:%M) Phase 0: oracle n-gram gate per grid" >> "$LOG"
for G in $GRIDS; do
  python3 -u -m mapformer.validate_miniworld --grids $G --T $T --n-episodes 40 --oracle \
    --out "$REPO/MINIWORLD_GRID${G}_GATES.md" >> "$LOG" 2>&1
  if grep -qE "\| G4 action n-gram \| FAIL" "$REPO/MINIWORLD_GRID${G}_GATES.md"; then
    echo "ABORT: grid $G oracle n-gram gate FAILED" >> "$LOG"; exit 1; fi
done
echo "$(date +%H:%M) gates PASS all grids" >> "$LOG"

# ---- Phase 1: pre-build buffers serially (fresh-map oracle) ----
echo "$(date +%H:%M) Phase 1: building buffers" >> "$LOG"
for G in $GRIDS; do
  for SEED in 0 1 2; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_buffer as B
B(W(grid_size=$G, seed=$SEED, oracle=True, fixed_map=False), $T, $NBUF, $SEED, n_workers=$NW)" >> "$LOG" 2>&1
  done
  for L in 512 1024; do
    python3 -c "
from mapformer.miniworld_env import MiniWorldWorld as W
from mapformer.train_miniworld import build_or_load_eval_buffer as E
E(W(grid_size=$G, seed=10000, oracle=True, fixed_map=False), $L, $ETRIALS, n_workers=$NW)" >> "$LOG" 2>&1
  done
done
echo "$(date +%H:%M) buffers ready; training 18 arms (<=2/GPU)" >> "$LOG"

# ---- Phase 2: 18 arms, strict <=MAXPG per GPU ----
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for G in $GRIDS; do
  for SEED in 0 1 2; do
    for V in $VARS; do
      OUT="$R/g${G}/s${SEED}"
      [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip g${G}_${V}_s${SEED}" >> "$LOG"; continue; }
      while :; do
        PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
        if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
        if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
        sleep 15
      done
      echo "$(date +%H:%M) g${G}_${V}_s${SEED} -> cuda:$GPU" >> "$LOG"
      python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
        --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
        --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
        --eval-trials $ETRIALS --eval-lengths 512 1024 --device "cuda:$GPU" \
        --output-dir "$OUT" > "$R/g${G}_${V}_s${SEED}.log" 2>&1 &
      PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
    done
  done
done
wait
echo "$(date +%H:%M) training exited; aggregating" >> "$LOG"

# ---- Phase 3: verify + per-grid Vanilla-RoPE effect (incl reused grid 8) ----
python3 -u - "$R" >> "$REPO/MINIWORLD_GRID_SWEEP.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]; REPO = os.path.dirname(R.rstrip('/').replace('/runs',''))
G8 = os.path.join(os.path.dirname(R), "miniworld_oracle")   # runs/miniworld_oracle
def acc(path, L):
    return json.load(open(path))[str(L)]["nb_acc"] if os.path.exists(path) else None
out = ["# MiniWorld grid sweep -- attention substitutability (fresh-map, oracle recode)", "",
       "Vanilla=path-int, RoPE=index. Prediction: path-int - index climbs from -0.53 "
       "(grid 8) toward positive as the grid grows and attention can no longer "
       "integrate position over the longer revisit distances. chance 0.0625.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "", "| grid | Vanilla (path-int) | RoPE (index) | effect (V-R) |",
            "|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        vs, rs = [], []
        for s in (0, 1, 2):
            if G == 8:
                vp = os.path.join(G8, f"s{s}", "Vanilla_oracle.json")
                rp = os.path.join(G8, f"s{s}", "RoPE_oracle.json")
            else:
                vp = os.path.join(R, f"g{G}", f"s{s}", "Vanilla_oracle.json")
                rp = os.path.join(R, f"g{G}", f"s{s}", "RoPE_oracle.json")
            av, ar = acc(vp, L), acc(rp, L)
            if av is not None: vs.append(av)
            if ar is not None: rs.append(ar)
        if vs and rs:
            eff = np.mean(vs) - np.mean(rs)
            out.append(f"| {G} | {np.mean(vs):.3f} | {np.mean(rs):.3f} | **{eff:+.3f}** "
                       f"(n={min(len(vs),len(rs))}) |")
        else:
            out.append(f"| {G} | — | — | (incomplete) |")
    out.append("")
print("\n".join(out))
PYEOF
touch "$REPO/.mw_grid_sweep_done"
echo "$(date) DONE" >> "$LOG"
