#!/usr/bin/env bash
# THE DECISIVE RUN. Nothing else in this environment is interpretable until it lands.
#
# THE QUESTION. Every MiniWorld comparison this session turned out to be mediated by
# convergence, not capability: held-out accuracy is an affine readout of final train
# loss (r = -0.996 over 57 runs), and RoPE NEVER converges at grid >= 16 (0/9, loss
# 0.47-0.84) yet is the subtrahend in every reported position effect. So the headline
# has two live explanations that the data cannot separate:
#   REPRESENTATIONAL -- index position genuinely cannot encode this task at grid 32
#   OPTIMISATION     -- RoPE just did not escape the plateau in 100 epochs
#
# THE TEST. Train RoPE at grid 32 to a GENUINELY FLAT loss and see where it lands.
#   RoPE reaches ~1.0            -> the crossover was OPTIMISATION. Headline dies.
#   RoPE plateaus flat and low   -> representational limit is supported, and THEN a
#                                   paired Vanilla batch is worth running.
# A positive answer needs only one seed (existence proof), so 3 seeds of RoPE alone
# is the cheapest decisive design. Vanilla is deliberately NOT run yet: if RoPE
# converges, the comparison is moot; if it does not, Vanilla must be trained in the
# SAME batch (rule 3) and this run tells us what budget that needs.
#
# WHY THE OLD BUDGET WAS NOT JUST "TOO SHORT". The default schedule is
# LinearLR(1.0 -> 0.0): LR decays from step one with no warmup. On a plateau-then-
# cliff loss landscape that is actively harmful -- by the time a run could escape the
# plateau its LR is ~0. So 100 epochs measured "did the transition fire early",
# not "can this model solve the task". Here: 4x the budget AND --schedule cosine
# (5% warmup, cosine decay to 10% of peak, so usable LR survives late training).
#
# CONVERGENCE IS REPORTED, NOT ASSUMED: the summary prints the loss slope over the
# final 10% of epochs. If that is not ~0 the run is STILL budget-limited and the
# answer is "unknown", not "representational".
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/rope_converge"; mkdir -p "$R"
LOG="$REPO/rope_converge.log"; echo "rope-converge queued $(date)" > "$LOG"
while [ ! -f "$REPO/.enwik8_long_done" ]; do sleep 120; done
echo "$(date +%H:%M) enwik8 finished; starting" >> "$LOG"

G=32; T=512; NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for SEED in 0 1 2; do
  OUT="$R/s${SEED}"
  [ -f "$OUT/RoPE_oracle.pt" ] && { echo "skip s$SEED" >> "$LOG"; continue; }
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) RoPE g32 s${SEED} (400ep, cosine) -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant RoPE --seed "$SEED" --oracle \
    --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
    --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
    --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/RoPE_s${SEED}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
done
wait
echo "$(date +%H:%M) done; summarising" >> "$LOG"

python3 -u - "$R" >> "$REPO/ROPE_CONVERGE.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np, torch
R = sys.argv[1]
out = ["# Does RoPE (index) solve grid 32 given enough optimisation?", "",
       "400 epochs (4x), warmup+cosine schedule, grid 32, oracle recode, n=3.",
       "Baseline for comparison -- the SAME arm at 100 epochs with linear decay:",
       "final loss 0.84 / 0.78 / 0.72, nb_acc 0.615 (mean).", "",
       "| seed | final loss | slope over last 10% | FLAT? | nb_acc T=512 | nb_acc T=1024 |",
       "|---|---|---|---|---|---|"]
losses, accs, flats = [], [], []
for s in (0, 1, 2):
    pt = os.path.join(R, f"s{s}", "RoPE_oracle.pt")
    js = os.path.join(R, f"s{s}", "RoPE_oracle.json")
    if not (os.path.exists(pt) and os.path.exists(js)):
        out.append(f"| {s} | — | — | — | — | missing |"); continue
    L = torch.load(pt, map_location="cpu")["losses"]
    r = json.load(open(js))
    tail = L[int(0.9 * len(L)):]
    slope = (tail[-1] - tail[0]) / max(1, len(tail) - 1)
    flat = abs(slope) < 5e-4
    losses.append(L[-1]); accs.append(r["512"]["nb_acc"]); flats.append(flat)
    out.append(f"| {s} | {L[-1]:.4f} | {slope:+.5f}/ep | {'YES' if flat else 'no'} | "
               f"{r['512']['nb_acc']:.3f} | {r['1024']['nb_acc']:.3f} |")
if losses:
    ml, ma = float(np.mean(losses)), float(np.mean(accs))
    out += ["", f"**mean final loss {ml:.4f}, mean nb_acc {ma:.3f}, "
                f"{sum(flats)}/{len(flats)} runs flat**", ""]
    if not all(flats):
        v = ("**STILL BUDGET-LIMITED.** Not all runs reached a flat loss, so the "
             "answer is UNKNOWN -- not 'representational'. Increase the budget again.")
    elif ma > 0.9:
        v = ("**THE CROSSOVER WAS OPTIMISATION.** RoPE solves grid 32 once trained to "
             "convergence, so 'index position cannot encode this task at scale' is "
             "false and the position-effect headline should be withdrawn entirely.")
    elif ma < 0.7:
        v = ("**REPRESENTATIONAL LIMIT SUPPORTED.** RoPE converged (flat loss) yet "
             "still fails. Next: train Vanilla in the SAME batch at this budget for a "
             "paired comparison -- that is the experiment that would rescue the claim.")
    else:
        v = "**AMBIGUOUS.** Converged but mid-range; a paired Vanilla batch is required."
    out += [v]
print("\n".join(out))
PYEOF
touch "$REPO/.rope_converge_done"
echo "$(date) DONE" >> "$LOG"
