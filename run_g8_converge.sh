#!/usr/bin/env bash
# THE MISSING HALF. Grid 8, both arms, converged -- completes (or kills) the crossover.
#
# STATE OF THE CLAIM. At grid 32 with both arms converged we now have a real position
# effect: path-int 0.927 vs index 0.754, +0.173, 3/3 sign-consistent, above the
# measured noise floor 0.150; path integration SOLVES the task (1.000 in 2/3 seeds)
# while index never exceeds 0.789 despite converging. That half is settled.
#
# The OTHER half has never been run at this budget. At grid 8 / 100 epochs / linear
# decay the numbers were RoPE 0.977 vs Vanilla 0.448 -- and BOTH arms were
# unconverged there (RoPE loss 0.17/0.17/0.23, Vanilla 0.93/1.14/1.14). Vanilla
# converged 0/3. So the -0.529 that anchors the entire "crossover" is an
# unconverged measurement, exactly like the grid-32 numbers were before yesterday.
#
# THE TEST: grid 8, Vanilla and RoPE, 400 epochs, warmup+cosine, n=3.
#   index still wins  -> the CROSSOVER IS REAL and now rests on converged data at
#                        BOTH ends: index wins on small maps, path-int on large ones.
#   path-int wins too -> there is NO crossover. Just a position effect that the
#                        100-epoch budget had INVERTED at grid 8, and every
#                        "crossover" statement in the repo should be withdrawn.
#   both ~equal       -> grid 8 does not discriminate; the effect is grid-32-only.
#
# Verdict thresholds are hard-coded against the MEASURED noise floor (0.150), and the
# script refuses to conclude anything unless all six arms reach a flat loss -- the
# failure mode that produced three retractions this session.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/g8_converge"; mkdir -p "$R"
LOG="$REPO/g8_converge.log"; echo "g8-converge start $(date)" > "$LOG"
G=8; T=512; NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do
    OUT="$R/s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip ${V}_s${SEED}" >> "$LOG"; continue; }
    while :; do
      PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
      if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
      sleep 30
    done
    echo "$(date +%H:%M) ${V} g8 s${SEED} (400ep, cosine) -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
      --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
      --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
      --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
      --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
    PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
  done
done
wait
echo "$(date +%H:%M) done; summarising" >> "$LOG"

python3 -u - "$R" >> "$REPO/CROSSOVER_CONVERGED.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np, torch
R = sys.argv[1]
def get(s, v):
    j = os.path.join(R, f"s{s}", f"{v}_oracle.json"); p = j.replace(".json", ".pt")
    if not (os.path.exists(j) and os.path.exists(p)): return None
    d = json.load(open(j)); L = torch.load(p, map_location="cpu")["losses"]
    tail = L[int(0.9*len(L)):]
    return dict(acc=d["512"]["nb_acc"], loss=L[-1],
                slope=(tail[-1]-tail[0])/max(1, len(tail)-1))
out = ["# Is the crossover real? Grid 8 with BOTH arms converged", "",
       "400 epochs, warmup + cosine, oracle recode, n=3 -- the same budget that "
       "settled grid 32. At 100 epochs grid 8 gave RoPE 0.977 vs Vanilla 0.448, but "
       "BOTH arms were unconverged there (Vanilla converged 0/3), so that -0.529 -- "
       "the anchor of the whole crossover claim -- was never a converged measurement.",
       "", "| seed | Vanilla loss | flat? | Vanilla acc | RoPE loss | flat? | RoPE acc | delta |",
       "|---|---|---|---|---|---|---|---|"]
ds, flats = [], []
for s in (0, 1, 2):
    a, b = get(s, "Vanilla"), get(s, "RoPE")
    if a is None or b is None:
        out.append(f"| {s} | — | — | — | — | — | — | missing |"); continue
    fa, fb = abs(a["slope"]) < 5e-4, abs(b["slope"]) < 5e-4
    flats += [fa, fb]; ds.append(a["acc"] - b["acc"])
    out.append(f"| {s} | {a['loss']:.4f} | {'Y' if fa else 'n'} | {a['acc']:.3f} | "
               f"{b['loss']:.4f} | {'Y' if fb else 'n'} | {b['acc']:.3f} | "
               f"**{a['acc']-b['acc']:+.3f}** |")
if ds:
    d = np.array(ds); m = d.mean()
    out += ["", f"**grid 8 converged effect (path-int − index): {m:+.3f}** "
                f"(sd {d.std(ddof=1):.3f}, {sum(flats)}/6 arms flat)",
            f"**grid 32 converged effect, for comparison: +0.173**", ""]
    if not all(flats):
        v = "**NOT ALL ARMS FLAT — still budget-limited. Answer unknown.**"
    elif m < -0.150:
        v = ("**THE CROSSOVER IS REAL.** Index wins at grid 8 and path integration wins "
             "at grid 32, both with every arm converged and both effects clearing the "
             "measured noise floor. The claim now rests on converged data at both ends "
             "-- which is the first time it has.")
    elif m > 0.150:
        v = ("**THERE IS NO CROSSOVER.** Path integration wins at grid 8 too once both "
             "arms converge, so the 100-epoch budget had INVERTED the sign. Every "
             "'crossover' statement in the repo should be withdrawn and replaced with a "
             "plain grid-independent position effect.")
    else:
        v = ("**GRID 8 DOES NOT DISCRIMINATE.** The arms land within the noise floor of "
             "each other, so there is no crossover -- only a grid-32 position effect.")
    out += [v]
print("\n".join(out))
PYEOF
touch "$REPO/.g8_converge_done"
echo "$(date) DONE" >> "$LOG"
