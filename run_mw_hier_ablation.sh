#!/usr/bin/env bash
# HIERARCHY ABLATION -- isolates POOLING, the variable the earlier comparison
# confounded. MapWM-Hier vs MapWM-FlatHG are the SAME Hourglass 3-block scaffold
# at IDENTICAL parameter count (2,384,026 both, verified), differing in exactly
# one bit: does the middle block see POOLED (k=2) tokens or full-resolution ones.
#
# WHY BOTH ARMS ARE RETRAINED. The prior claim ("hierarchy adds +0.283 to path
# integration") compared MapWM-Hier against Vanilla -- different scaffold, different
# depth, 3.17M vs 2.38M params -- AND across batches. Retraining MapWM-Hier here
# alongside FlatHG puts the whole comparison in ONE batch, so no reproducibility
# control or cross-batch caveat is needed: it is within-batch by construction.
# (Rule 3: never compare a fresh arm to a stored one.)
#
# Grids 8/16/24/32 x 2 variants x 3 seeds = 24 arms, fresh-map oracle recode,
# 100 epochs. Buffers all cached from the flat + hier sweeps. Both GPUs, <=2/GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_hier_ablation"; mkdir -p "$R"
LOG="$REPO/miniworld_hier_ablation.log"; echo "hier ablation start $(date)" > "$LOG"
GRIDS="8 16 24 32"; VARS="MapWM-Hier MapWM-FlatHG"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

echo "$(date +%H:%M) buffers (cached from prior sweeps)" >> "$LOG"
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
echo "$(date +%H:%M) training exited; verifying" >> "$LOG"

MISS=0
for G in $GRIDS; do for SEED in 0 1 2; do for V in $VARS; do
  [ -f "$R/g${G}/s${SEED}/${V}_oracle.json" ] || { echo "MISSING g${G}_${V}_s${SEED}" >> "$LOG"; MISS=$((MISS+1)); }
done; done; done
echo "missing: $MISS" >> "$LOG"

python3 -u - "$R" >> "$REPO/MINIWORLD_HIER_ABLATION.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]
def acc(g, s, v, L):
    p = os.path.join(R, f"g{g}", f"s{s}", f"{v}_oracle.json")
    return json.load(open(p))[str(L)]["nb_acc"] if os.path.exists(p) else None
out = ["# MiniWorld — HIERARCHY ABLATION (isolates pooling)", "",
       "MapWM-Hier vs MapWM-FlatHG: the SAME Hourglass 3-block scaffold at IDENTICAL "
       "parameter count (2,384,026 both), differing ONLY in whether the middle block "
       "sees pooled (k=2) or full-resolution tokens. Both retrained in ONE batch, so "
       "this is within-batch by construction. Fresh-map, oracle recode, n=3, "
       "chance 0.0625.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "", "| grid | MapWM-Hier (pooled) | MapWM-FlatHG (no pooling) | effect of POOLING |",
            "|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        h = [acc(G, s, "MapWM-Hier", L) for s in (0, 1, 2)]
        f = [acc(G, s, "MapWM-FlatHG", L) for s in (0, 1, 2)]
        h = [x for x in h if x is not None]; f = [x for x in f if x is not None]
        if h and f:
            per = ", ".join(f"{a-b:+.3f}" for a, b in zip(h, f)) if len(h) == len(f) else ""
            out.append(f"| {G} | {np.mean(h):.3f} | {np.mean(f):.3f} | "
                       f"**{np.mean(h)-np.mean(f):+.3f}** (n={min(len(h),len(f))}; {per}) |")
        else:
            out.append(f"| {G} | — | — | (incomplete) |")
    out.append("")
out += ["> Read this table INSTEAD of the earlier MapWM-Hier vs Vanilla comparison,",
        "> which differed in scaffold, depth and parameter count (2.38M vs 3.17M) and",
        "> ran across two batches. Only the numbers here isolate pooling."]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_hier_abl_done"
echo "$(date) DONE (miss=$MISS)" >> "$LOG"
