#!/usr/bin/env bash
# SEED-0 PROBE of three new arms across the grid axis. Deliberately ONE SEED:
# project rule -- "run one seed of everything first; outer loop seed, inner loop
# variant: lands a full low-confidence table fastest, then tighten error bars
# after". 3 variants x 4 grids x 1 seed = 12 arms (~3h on both GPUs) instead of
# the 36-arm/9h full-seed version, which was over-scoped for questions this
# directional.
#
#   NoPE      -- no rotation at all (param-identical to RoPE). Is index-RoPE
#                actually better than NOTHING on a spatial task, or is rotating
#                by an ordinal angle a handicap that made our control a straw man?
#                The navigation literature has no NoPE arm at all.
#   ConvDelta -- SRoPE's conv1d before the cumsum. cumsum and a difference filter
#                are inverses, so it learns HOW MUCH to accumulate. Prediction:
#                unnecessary here (navigation position IS the full integral).
#                Identity-init => exactly Vanilla at step 0.
#   GateDelta -- SRoPE's sigmoid gate on Delta. Our stream alternates
#                [action, obs, ...]; only actions displace. Prediction: helps, or
#                at least converges faster. +1.0% params (mild confound).
#
# Same config as run_mw_grid_sweep.sh so these slot into that table beside
# Vanilla / RoPE. Buffers cached. READ per-seed loss before trusting anything --
# the hierarchy ablation showed means here can be pure convergence artifacts.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"
LOG="$REPO/miniworld_probe3.log"; echo "probe3 start $(date)" > "$LOG"
GRIDS="8 16 24 32"; VARS="NoPE ConvDelta GateDelta"; SEED=0
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for G in $GRIDS; do
  for V in $VARS; do
    OUT="$R/g${G}/s${SEED}"
    [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip g${G}_${V}" >> "$LOG"; continue; }
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
wait
echo "$(date +%H:%M) done; aggregating" >> "$LOG"

python3 -u - "$R" >> "$REPO/MINIWORLD_PROBE3.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
R = sys.argv[1]
def get(g, v, L):
    p = os.path.join(R, f"g{g}", "s0", f"{v}_oracle.json")
    return json.load(open(p))[str(L)]["nb_acc"] if os.path.exists(p) else None
def loss(g, v):
    p = os.path.join(R, f"g{g}", "s0", f"{v}_oracle.pt")
    if not os.path.exists(p): return None
    import torch
    return torch.load(p, map_location="cpu")["losses"][-1]
out = ["# MiniWorld — seed-0 probe: NoPE, ConvDelta, GateDelta", "",
       "**n=1, low confidence by design** — lands the table shape first, seeds after. "
       "Final train loss shown beside every accuracy: the hierarchy ablation showed "
       "differences here are often convergence, not capability.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "",
            "| grid | Vanilla | RoPE | NoPE | ConvDelta | GateDelta |",
            "|---|---|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        cells = []
        for v in ("Vanilla", "RoPE", "NoPE", "ConvDelta", "GateDelta"):
            a, l = get(G, v, L), loss(G, v)
            cells.append("—" if a is None else
                         f"{a:.3f}" + (f" <sub>({l:.2f})</sub>" if l is not None else ""))
        out.append(f"| {G} | " + " | ".join(cells) + " |")
    out.append("")
out += ["> Values are nb_acc with (final train loss) beneath. Compare NoPE vs RoPE",
        "> (is the ordinal rotation a handicap?), ConvDelta vs Vanilla (does",
        "> learning how-much-to-accumulate help when full accumulation is already",
        "> correct?), GateDelta vs Vanilla (does an explicit action/obs gate help?)."]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_probe3_done"
echo "$(date) DONE" >> "$LOG"
