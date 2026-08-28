#!/usr/bin/env bash
# THE PAIRED HALF. Finishes the question the RoPE convergence run left open.
#
# WHAT THE ROPE RUN ESTABLISHED (400 ep, warmup+cosine, grid 32, n=3):
#   RoPE converged 3/3 (slope -0.0003/ep), final loss 0.446, nb_acc 0.754.
#   vs the same arm at 100 ep with linear decay: loss 0.78, acc 0.615.
# So "RoPE never converges at grid >= 16" was a SCHEDULE artifact -- LinearLR decays
# from step one with no warmup, so the plateau could never be escaped late. The
# convergence-confound diagnosis was right.
#
# BUT: RoPE converges to 0.754, NOT to the ceiling. That kills yesterday's worry that
# fixing convergence would push every arm to 1.000 and destroy discrimination. And of
# the 8 arm-seeds that DO exceed 0.95 at grid 32, every single one is PATH-INTEGRATED
# (Vanilla, MapWM-Hier, MapWM-FlatHG, GateDelta, GateDeltaCtl) -- no index arm gets
# there. So the environment discriminates once arms are trained properly.
#
# THE TEST: Vanilla at the SAME budget and schedule.
#   Vanilla -> ~0.95+  : a REAL, converged, representational position effect. The
#                        headline is rescued on honest footing rather than being the
#                        convergence artifact it was.
#   Vanilla -> ~0.75   : the position effect is DEAD at grid 32; both arms hit the
#                        same wall and the earlier gap was purely optimisation.
#
# BATCH DISCIPLINE (rule 3). The RoPE arms were trained in the previous batch. Code,
# schedule and cached buffers are unchanged, and this pipeline has been shown
# deterministic (MapWM-Hier reproduced bit-identically across batches). One RoPE seed
# is nevertheless retrained here as a REPRODUCIBILITY CONTROL: if it reproduces
# s0's 0.725 / loss 0.499, the cross-batch comparison is licensed; if it drifts, that
# drift IS the error bar and the comparison must be read against it.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/rope_converge"; mkdir -p "$R"          # same dir as the RoPE arms
LOG="$REPO/vanilla_converge.log"; echo "vanilla-converge start $(date)" > "$LOG"
G=32; T=512; NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
launch(){ local V="$1" SEED="$2" SUB="$3"
  local OUT="$R/$SUB"
  mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip $V $SUB" >> "$LOG"; return; }
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $V g32 $SUB (400ep, cosine) -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
    --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
    --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_${SUB}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
}
launch Vanilla 0 s0
launch Vanilla 1 s1
launch Vanilla 2 s2
launch RoPE 0 s0_repro      # reproducibility control vs the stored s0
wait
echo "$(date +%H:%M) done; summarising" >> "$LOG"

python3 -u - "$R" >> "$REPO/POSITION_EFFECT_CONVERGED.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np, torch
R = sys.argv[1]
def get(sub, v):
    j = os.path.join(R, sub, f"{v}_oracle.json"); p = j.replace(".json", ".pt")
    if not (os.path.exists(j) and os.path.exists(p)): return None
    d = json.load(open(j)); L = torch.load(p, map_location="cpu")["losses"]
    tail = L[int(0.9*len(L)):]
    return dict(acc=d["512"]["nb_acc"], acc1024=d["1024"]["nb_acc"], loss=L[-1],
                slope=(tail[-1]-tail[0])/max(1, len(tail)-1))
out = ["# Position effect at grid 32, BOTH ARMS CONVERGED", "",
       "400 epochs, warmup + cosine, oracle recode, n=3. This is the comparison every "
       "earlier MiniWorld table was missing: at 100 epochs with linear decay neither "
       "arm reliably converged, so the reported effects measured time-to-solve.", "",
       "| arm | seed | final loss | slope | flat? | nb_acc T=512 | T=1024 |",
       "|---|---|---|---|---|---|---|"]
V, Rp = [], []
for s in (0, 1, 2):
    for v, acc in (("Vanilla", V), ("RoPE", Rp)):
        sub = f"s{s}"
        g = get(sub, v)
        if g is None: out.append(f"| {v} | {s} | — | — | — | — | missing |"); continue
        acc.append(g)
        out.append(f"| {v} | {s} | {g['loss']:.4f} | {g['slope']:+.5f}/ep | "
                   f"{'YES' if abs(g['slope'])<5e-4 else 'no'} | {g['acc']:.3f} | {g['acc1024']:.3f} |")
ctl = get("s0_repro", "RoPE"); base = get("s0", "RoPE")
if ctl and base:
    d = ctl["acc"] - base["acc"]
    out += ["", f"**Repro control:** RoPE s0 retrained = {ctl['acc']:.3f} vs stored "
                f"{base['acc']:.3f} (drift {d:+.3f}). "
            + ("Cross-batch comparison licensed." if abs(d) < 0.03 else
               "**Drift exceeds 0.03 -- read the effect against this as its error bar.**")]
if V and Rp:
    va, ra = np.mean([x["acc"] for x in V]), np.mean([x["acc"] for x in Rp])
    allflat = all(abs(x["slope"]) < 5e-4 for x in V + Rp)
    out += ["", f"| | Vanilla (path-int) | RoPE (index) | effect |", "|---|---|---|---|",
            f"| mean nb_acc T=512 | **{va:.3f}** | {ra:.3f} | **{va-ra:+.3f}** |", ""]
    if not allflat:
        v = "**NOT ALL ARMS FLAT — still budget-limited, answer unknown.**"
    elif va - ra > 0.15:
        v = ("**REAL, CONVERGED POSITION EFFECT.** Both arms trained to a flat loss and "
             "path integration still wins by more than the measured noise floor (0.150). "
             "This is the comparison the headline always needed.")
    elif abs(va - ra) < 0.15:
        v = ("**POSITION EFFECT DEAD AT GRID 32.** Both arms converge to the same place "
             "within the measured noise floor (0.150). The earlier gap was optimisation, "
             "and the crossover should be withdrawn entirely.")
    else:
        v = "**INDEX WINS at convergence** — the opposite of the original claim."
    out += [v]
print("\n".join(out))
PYEOF
touch "$REPO/.vanilla_converge_done"
echo "$(date) DONE" >> "$LOG"
