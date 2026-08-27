#!/usr/bin/env bash
# NoPE arm for the MiniWorld grid sweep -- the null hypothesis the navigation side
# of this literature entirely lacks (MapFormer has ZERO NoPE baselines; Selective
# RoPE runs NoPE everywhere and it SOMETIMES WINS: GLA 1.3B avg acc NoPE 55.2 >
# SRoPE 54.6 > RoPE 54.4).
#
# WHAT IT SETTLES. Our index arms do not merely lack useful position -- they rotate
# q/k by an ORDINAL angle, the wrong signal for a spatial map. MiniWorld already
# showed a confidently-wrong position code can be WORSE than a degenerate one (the
# 24-bin allocentric recode scored below raw). If NoPE > RoPE here, our index
# baseline is a straw man and part of the measured position effect is that
# handicap. NoPE is param-identical to RoPE (3,172,890 both) with the rotation set
# to identity, so it isolates the positional signal and nothing else.
#
# Grids 8/16/24/32 x 3 seeds = 12 arms, fresh-map oracle recode, 100 epochs,
# same config as run_mw_grid_sweep.sh so it slots into that table. Buffers cached.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"          # same dir as the flat sweep
LOG="$REPO/miniworld_nope.log"; echo "nope sweep start $(date)" > "$LOG"
GRIDS="8 16 24 32"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2

declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for G in $GRIDS; do
  for SEED in 0 1 2; do
    OUT="$R/g${G}/s${SEED}"
    [ -f "$OUT/NoPE_oracle.pt" ] && { echo "skip g${G}_s${SEED}" >> "$LOG"; continue; }
    while :; do
      PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
      if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
      sleep 15
    done
    echo "$(date +%H:%M) g${G}_NoPE_s${SEED} -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_miniworld --variant NoPE --seed "$SEED" --oracle \
      --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
      --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
      --eval-trials $ETRIALS --eval-lengths 512 1024 --device "cuda:$GPU" \
      --output-dir "$OUT" > "$R/g${G}_NoPE_s${SEED}.log" 2>&1 &
    PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
  done
done
wait
echo "$(date +%H:%M) training exited; aggregating 3-way" >> "$LOG"

python3 -u - "$R" >> "$REPO/MINIWORLD_NOPE.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]
def a(g, s, v, L):
    p = os.path.join(R, f"g{g}", f"s{s}", f"{v}_oracle.json")
    return json.load(open(p))[str(L)]["nb_acc"] if os.path.exists(p) else None
def col(g, v, L):
    xs = [a(g, s, v, L) for s in (0, 1, 2)]
    xs = [x for x in xs if x is not None]
    return (np.mean(xs), len(xs)) if xs else (None, 0)
out = ["# MiniWorld grid sweep — with the NoPE null arm", "",
       "Vanilla = path-integrated; RoPE = index (ordinal rotation); **NoPE = no "
       "rotation at all** (param-identical to RoPE, 3,172,890 both). Fresh-map, "
       "oracle recode, n=3, chance 0.0625.", "",
       "Key question: is index-RoPE actually BETTER than no position code, or is "
       "rotating by an ordinal angle actively harmful on a spatial task? If "
       "NoPE > RoPE, our index baseline is a straw man.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "",
            "| grid | Vanilla (path-int) | RoPE (index) | NoPE (none) | RoPE − NoPE | path-int − best-index |",
            "|---|---|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        v, _ = col(G, "Vanilla", L); r, _ = col(G, "RoPE", L); n, nn = col(G, "NoPE", L)
        if v is None or r is None or n is None:
            out.append(f"| {G} | — | — | — | — | (incomplete) |"); continue
        best_ix = max(r, n)
        out.append(f"| {G} | {v:.3f} | {r:.3f} | {n:.3f} | **{r-n:+.3f}** | "
                   f"**{v-best_ix:+.3f}** |")
    out.append("")
out += ["> `RoPE − NoPE` > 0 means the ordinal rotation helps; < 0 means it is a",
        "> handicap and our index control was weaker than it needed to be.",
        "> `path-int − best-index` re-measures the position effect against the",
        "> STRONGER of the two index arms, which is the honest comparison."]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_nope_done"
echo "$(date) DONE" >> "$LOG"
