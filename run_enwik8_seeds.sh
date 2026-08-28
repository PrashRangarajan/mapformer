#!/usr/bin/env bash
# Seeds 1,2 for the two headline enwik8 arms -- gives n=3 on the composition claim.
#
# WHY ONLY TWO ARMS. The interesting result is whether MapPoPE-Flat (path-int + PoPE)
# beats plain RoPE, i.e. whether the two orthogonal mechanisms COMPOSE on language as
# they do on navigation. That needs RoPE and MapPoPE-Flat_r4 at n=3. Reseeding all
# four arms would cost 2x for error bars on the intermediate rows, which are not the
# claim. (PoPE-Flat and Vanilla_r4 stay at n=1 and should be reported as such.)
#
# WHAT THE 36k RUN SHOWED (n=1, mean of last 5 checkpoints):
#   MapPoPE-Flat_r4 1.3786  (-0.0078 vs RoPE)   <- best
#   PoPE-Flat       1.3806  (-0.0058)
#   Vanilla_r4      1.3841  (-0.0022)
#   RoPE            1.3864  (baseline)
# Ordering is STABLE under smoothing (final and mean-of-last-5 agree), unlike the 12k
# run where smoothing INVERTED it. The deterministic-eval fix cut checkpoint sd from
# 0.02-0.07 to 0.003-0.007. But the sd above is WITHIN-RUN checkpoint noise, not seed
# variance -- which is exactly what this run supplies.
#
# HONEST BAR: the effect is -0.0078 against a within-run sd of ~0.005. If the three
# seeds do not agree in sign, the composition claim is not supported at this scale and
# should be reported as such rather than as a trend.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
OUT="$REPO/enwik8_long"; mkdir -p "$OUT"
LOG="$REPO/enwik8_seeds.log"; echo "enwik8 seeds queued $(date)" > "$LOG"
while [ ! -f "$REPO/.rope_converge_done" ]; do sleep 120; done
echo "$(date +%H:%M) rope-converge finished; starting" >> "$LOG"

SEQ=512; BS=16; ITERS=36000; LR=2e-4; MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
run(){ local NAME="$1" R="$2" TAG="$3" SEED="$4"
  local F="$OUT/${NAME}${TAG}.json"
  if [ -f "$F" ] && python3 -c "import json,sys; sys.exit(0 if 'wall_total_s' in json.load(open(sys.argv[1])) else 1)" "$F" 2>/dev/null; then
    echo "skip ${NAME}${TAG} (complete)" >> "$LOG"; return; fi
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 20
  done
  echo "$(date +%H:%M) ${NAME}${TAG} seed $SEED -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_hourglass_enwik8 --model "$NAME" --tag "$TAG" --seed "$SEED" \
    --seq-len $SEQ --batch-size $BS --iters $ITERS --lr $LR --eval-every 1000 \
    --dim 512 --heads 8 --n-layers 9 --bottleneck-r "$R" \
    --out "$OUT" --device "cuda:$GPU" > "$OUT/${NAME}${TAG}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 5
}
run RoPE 2 "_s1" 1
run MapPoPE-Flat 4 "_r4_s1" 1
run RoPE 2 "_s2" 2
run MapPoPE-Flat 4 "_r4_s2" 2
wait
echo "$(date +%H:%M) done; aggregating n=3" >> "$LOG"

python3 -u - "$OUT" >> "$REPO/ENWIK8_SEEDS.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R = sys.argv[1]
def last5(fn):
    p = os.path.join(R, fn)
    if not os.path.exists(p): return None
    c = json.load(open(p))["curve"]
    return float(np.mean([x["val_bpc"] for x in c[-5:]]))
pairs = [("RoPE.json", "MapPoPE-Flat_r4.json", 0),
         ("RoPE_s1.json", "MapPoPE-Flat_r4_s1.json", 1),
         ("RoPE_s2.json", "MapPoPE-Flat_r4_s2.json", 2)]
out = ["# enwik8 — does the PoPE x path-integration combination beat RoPE? (n=3)", "",
       "36k iters, seq 512, deterministic val (fixed generator -- identical batches for "
       "every arm and checkpoint), r=4 for the path-integrating arm, param-matched to "
       "0.03%. Values are the mean of the last 5 checkpoints. **Lower is better.**", "",
       "| seed | RoPE | MapPoPE-Flat r4 | delta |", "|---|---|---|---|"]
ds = []
for a, b, s in pairs:
    va, vb = last5(a), last5(b)
    if va is None or vb is None:
        out.append(f"| {s} | — | — | missing |"); continue
    ds.append(vb - va)
    out.append(f"| {s} | {va:.4f} | {vb:.4f} | **{vb-va:+.4f}** |")
if ds:
    d = np.array(ds)
    out += ["", f"- mean **{d.mean():+.4f}**"
            + (f", sd {d.std(ddof=1):.4f}" if len(d) > 1 else ""),
            f"- sign-consistent: **{'YES' if (d < 0).all() or (d > 0).all() else 'NO'}** "
            f"({int((d<0).sum())}/{len(d)} favour the combination)", ""]
    if len(d) > 1 and (d < 0).all():
        out += ["> All seeds favour the combination. At this scale (295M tokens vs the",
                "> paper's 100B) the effect is small but consistent -- the two orthogonal",
                "> mechanisms compose on language, as they do on navigation."]
    else:
        out += ["> Not sign-consistent across seeds: the composition claim is NOT",
                "> supported at this scale and should be reported as unmeasured."]
print("\n".join(out))
PYEOF
touch "$REPO/.enwik8_seeds_done"
echo "$(date) DONE" >> "$LOG"
