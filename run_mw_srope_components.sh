#!/usr/bin/env bash
# The two Selective RoPE components MapFormer lacks, tested on navigation.
# Neither has ever been evaluated in the other paper's domain.
#
# ConvDelta -- SRoPE's conv1d on Delta before the cumsum. Because cumsum and a
#   difference filter are inverses, the conv LEARNS HOW MUCH TO ACCUMULATE:
#   identity kernel -> full accumulation; [1,-1] -> none (position = current
#   content). PREDICTION: navigation position IS the integral of displacement, so
#   full accumulation is already correct and the conv should be UNNECESSARY --
#   the model should keep it near the identity. Verified EXACTLY equal to Vanilla
#   at init (max|diff| = 0.00e+00), so any deviation is learned.
#
# GateDelta -- SRoPE's sigmoid gate on Delta (its best-performing addition on MAD).
#   Our stream alternates [action, obs, ...] and only ACTIONS should displace the
#   agent; MapFormer must learn Delta~=0 for observations implicitly through the
#   rank-2 bottleneck. PREDICTION: the explicit gate makes that free and should
#   help, or at minimum converge faster. Caveat: +32,896 params (+1.0%) -- a mild
#   capacity confound, unlike ConvDelta's +384 (+0.01%).
#
# Same config as run_mw_grid_sweep.sh so these slot into that table alongside
# Vanilla / RoPE / NoPE: grids 8/16/24/32 x 3 seeds x 2 variants = 24 arms,
# fresh-map oracle recode, 100 epochs. Buffers cached.
#
# WAITS for the NoPE sweep to finish so the GPUs are not oversubscribed.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"
LOG="$REPO/miniworld_srope_comp.log"; echo "srope-components queued $(date)" > "$LOG"

while [ ! -f "$REPO/.mw_nope_done" ]; do sleep 60; done
echo "$(date +%H:%M) NoPE finished; starting" >> "$LOG"

GRIDS="8 16 24 32"; VARS="ConvDelta GateDelta"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
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
echo "$(date +%H:%M) training exited; aggregating 5-way" >> "$LOG"

python3 -u - "$R" >> "$REPO/MINIWORLD_SROPE_COMPONENTS.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]
def col(g, v, L):
    xs = []
    for s in (0, 1, 2):
        p = os.path.join(R, f"g{g}", f"s{s}", f"{v}_oracle.json")
        if os.path.exists(p): xs.append(json.load(open(p))[str(L)]["nb_acc"])
    return (np.mean(xs), len(xs)) if xs else (None, 0)
out = ["# MiniWorld — Selective RoPE components ported to navigation", "",
       "Selective RoPE (ICLR 2026) and MapFormer were posted 3 days apart with the "
       "same primitive; neither cites the other. These are the two components SRoPE "
       "has and MapFormer lacks, tested here for the first time on navigation. "
       "Fresh-map, oracle recode, n=3, chance 0.0625.", "",
       "**ConvDelta** = conv1d on Delta before the cumsum (learns HOW MUCH to "
       "accumulate; identity-init, so exactly Vanilla at step 0). Prediction: "
       "unnecessary here, since navigation position IS the full integral.", "",
       "**GateDelta** = sigmoid gate on Delta (only actions should displace the "
       "agent; MapFormer must learn that implicitly). +1.0% params — mild confound.", ""]
for L in (512, 1024):
    out += [f"## T={L}", "",
            "| grid | Vanilla | ConvDelta | GateDelta | Conv − Vanilla | Gate − Vanilla |",
            "|---|---|---|---|---|---|"]
    for G in (8, 16, 24, 32):
        v, _ = col(G, "Vanilla", L); c, _ = col(G, "ConvDelta", L); g, _ = col(G, "GateDelta", L)
        if v is None or c is None or g is None:
            out.append(f"| {G} | — | — | — | — | (incomplete) |"); continue
        out.append(f"| {G} | {v:.3f} | {c:.3f} | {g:.3f} | **{c-v:+.3f}** | **{g-v:+.3f}** |")
    out.append("")
out += ["> Read per-seed spread and final training loss before trusting any mean —",
        "> the hierarchy ablation showed these means can be outlier-driven, with the",
        "> real effect sitting on convergence rather than capability."]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_srope_comp_done"
echo "$(date) DONE" >> "$LOG"
