#!/usr/bin/env bash
# Seeds 1,2 for ConvDelta and GateDelta at grids 16/24/32 -- tightening the seed-0
# probe. 2 variants x 3 grids x 2 seeds = 12 arms (~3h). Vanilla/RoPE seeds already
# exist from run_mw_grid_sweep.sh, so this yields matched 3-seed comparisons.
#
# WHAT IT MUST SETTLE. The seed-0 probe showed ConvDelta (0.969 at g24) and
# GateDelta (0.745) beating Vanilla (0.623) -- but Vanilla's final train loss there
# was 0.76 vs ConvDelta's 0.10, i.e. VANILLA DID NOT CONVERGE. The hierarchy
# ablation showed exactly this shape, and once conditioned on both arms converging
# the effect was EXACTLY ZERO. Also, Vanilla seed 0 is a bad draw: its 3-seed means
# at g24/g32 are 0.694/0.703 vs the 0.623/0.559 seen at seed 0.
#
# So the aggregator below does the thing that matters: report per-seed accuracy
# WITH final train loss, and compute the effect BOTH pooled and restricted to
# pairs where BOTH arms converged (loss < 0.4). If the advantage survives the
# restricted comparison it is real -- and it would be a component from the LANGUAGE
# paper (Selective RoPE) improving the NAVIGATION model, the cross-domain transfer
# neither paper tested. If it evaporates, it is trainability, like hierarchy was.
#
# Grid 8 omitted: every arm is near the floor there and it adds nothing.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"
LOG="$REPO/miniworld_probe3_seeds.log"; echo "probe3-seeds start $(date)" > "$LOG"
GRIDS="16 24 32"; VARS="ConvDelta GateDelta"; SEEDS="1 2"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for G in $GRIDS; do
  for SEED in $SEEDS; do
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
echo "$(date +%H:%M) done; aggregating with convergence conditioning" >> "$LOG"

python3 -u - "$R" >> "$REPO/MINIWORLD_SROPE_COMPONENTS.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np, torch
R = sys.argv[1]; THRESH = 0.4
def cell(g, s, v, L):
    j = os.path.join(R, f"g{g}", f"s{s}", f"{v}_oracle.json")
    p = os.path.join(R, f"g{g}", f"s{s}", f"{v}_oracle.pt")
    if not (os.path.exists(j) and os.path.exists(p)): return None, None
    return (json.load(open(j))[str(L)]["nb_acc"],
            torch.load(p, map_location="cpu")["losses"][-1])
out = ["# Selective RoPE components on navigation — 3 seeds, convergence-conditioned", "",
       "ConvDelta (SRoPE's conv1d before the cumsum) and GateDelta (SRoPE's sigmoid "
       "gate on Delta) vs Vanilla MapFormer. Fresh-map, oracle recode, grids 16/24/32, "
       "n=3. **Accuracy (final train loss)** — the hierarchy ablation showed effects "
       "here are frequently convergence, not capability, so the conditioned row is "
       "the one that counts.", ""]
for L in (512, 1024):
    out += [f"## T={L}", ""]
    for V in ("ConvDelta", "GateDelta"):
        out += [f"### {V} vs Vanilla", "",
                "| grid | seed | Vanilla | " + V + " | delta | both converged? |",
                "|---|---|---|---|---|---|"]
        pooled, restricted = [], []
        for G in (16, 24, 32):
            for s in (0, 1, 2):
                av, al = cell(G, s, "Vanilla", L); bv, bl = cell(G, s, V, L)
                if av is None or bv is None:
                    out.append(f"| {G} | {s} | — | — | — | missing |"); continue
                d = bv - av; pooled.append(d)
                ok = (al < THRESH) and (bl < THRESH)
                if ok: restricted.append(d)
                out.append(f"| {G} | {s} | {av:.3f} <sub>({al:.2f})</sub> | "
                           f"{bv:.3f} <sub>({bl:.2f})</sub> | {d:+.3f} | "
                           f"{'YES' if ok else 'no'} |")
        pm = np.mean(pooled) if pooled else float('nan')
        rm = np.mean(restricted) if restricted else float('nan')
        out += ["",
                f"- **pooled** ({len(pooled)} pairs): **{pm:+.3f}**",
                f"- **both-converged only** ({len(restricted)} pairs): "
                + (f"**{rm:+.3f}**" if restricted else "*no pair had both arms converge*"),
                ""]
out += ["> If the both-converged effect is ~0, this is trainability (as hierarchy was).",
        "> If it survives, a component from the LANGUAGE paper improves the NAVIGATION",
        "> model — the cross-domain transfer neither paper tested."]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_probe3_seeds_done"
echo "$(date) DONE" >> "$LOG"
