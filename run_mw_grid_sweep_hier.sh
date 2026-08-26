#!/usr/bin/env bash
# HIERARCHY grid sweep -- does a time-hierarchy EXTEND the attention horizon, so
# the index model substitutes over longer distances and the crossover grid moves
# UP? Mirror of run_mw_grid_sweep.sh with the Hourglass (k=2 pooling) variants:
# {MapWM-Hier=path-int+hier, Plain-Hier=index+hier}, both 2.38M params (internally
# matched), vs the flat pair {Vanilla,RoPE} 3.17M (internally matched). Compare the
# per-pair crossover GRID: if the hier pair crosses at a HIGHER grid than the flat
# pair, hierarchy let index keep up longer. Fresh-map, oracle recode, grids
# {8,16,24,32} x 2 variants x 3 seeds = 24 arms, 100 epochs. Buffers reuse the flat
# sweep's oracle buffers (variant-independent) + grid-8 from the oracle run.
# NOT auto-launched -- run AFTER run_mw_grid_sweep.sh completes.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep_hier"; mkdir -p "$R"
LOG="$REPO/miniworld_grid_sweep_hier.log"; echo "hier grid sweep start $(date)" > "$LOG"
GRIDS="8 16 24 32"; VARS="MapWM-Hier Plain-Hier"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

# ---- Phase 1: ensure buffers exist (all cached from the flat sweep / oracle run;
#      grid 8 oracle fresh buffer is in the cache from the oracle experiment) ----
echo "$(date +%H:%M) Phase 1: ensuring oracle fresh-map buffers (cached)" >> "$LOG"
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
echo "$(date +%H:%M) buffers ready; training 24 arms (<=2/GPU)" >> "$LOG"

# ---- Phase 2: 24 arms, strict <=MAXPG per GPU ----
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

# ---- Phase 3: per-grid MapWM-Hier - Plain-Hier effect ----
python3 -u - "$R" >> "$REPO/MINIWORLD_GRID_SWEEP_HIER.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]
def acc(path, L):
    return json.load(open(path))[str(L)]["nb_acc"] if os.path.exists(path) else None
out = ["# MiniWorld grid sweep -- HIERARCHY (Hourglass k=2), fresh-map oracle recode", "",
       "MapWM-Hier=path-int+hier, Plain-Hier=index+hier (both 2.38M, internally "
       "matched). Question: does the hier-pair crossover grid sit HIGHER than the "
       "flat-pair (Vanilla-RoPE) crossover, i.e. does hierarchy let index substitute "
       "over longer distances? chance 0.0625.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "", "| grid | MapWM-Hier (path-int) | Plain-Hier (index) | effect (H-I) |",
            "|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        vs, rs = [], []
        for s in (0, 1, 2):
            av = acc(os.path.join(R, f"g{G}", f"s{s}", "MapWM-Hier_oracle.json"), L)
            ar = acc(os.path.join(R, f"g{G}", f"s{s}", "Plain-Hier_oracle.json"), L)
            if av is not None: vs.append(av)
            if ar is not None: rs.append(ar)
        if vs and rs:
            out.append(f"| {G} | {np.mean(vs):.3f} | {np.mean(rs):.3f} | "
                       f"**{np.mean(vs)-np.mean(rs):+.3f}** (n={min(len(vs),len(rs))}) |")
        else:
            out.append(f"| {G} | — | — | (incomplete) |")
    out.append("")
print("\n".join(out))
PYEOF
touch "$REPO/.mw_grid_sweep_hier_done"
echo "$(date) DONE" >> "$LOG"
