#!/usr/bin/env bash
# enwik8 rerun fixing BOTH caveats found in the 12k run.
#
# CAVEAT 1 (budget). At 12k iters EVERY arm was still improving, and the remaining
# per-checkpoint gain (MapPoPE-Flat -0.061 over the last 2k) was LARGER than the
# entire spread between models (0.03). MapPoPE-Flat went from worst-but-one at 10k
# to best at 12k. So the 12k ordering is not stable -- standing rule 5. Here: 36k
# iters, 3x the budget.
#
# CAVEAT 2 (rank). We ran bottleneck_r=2; MapFormer's OWN OpenWebText run uses
# r=4, and rank is decisive in their navigation data (2D: r=1 -> 0.66, r=2 -> 1.00).
# So the path-integration arms may have been under-configured. Here: r=4 for the
# path-integrating arms, matching the paper's language setting.
#
# NOT a caveat, RESOLVED: the apparent train/val anomaly was a measurement artifact
# -- train_bpc is a single-minibatch estimate and oscillates +/-0.2, while val is
# averaged. No overfitting, no divergent optimisation dynamics.
#
# Arms (param-matched to 0.03%): RoPE (baseline, index+RoPE) / Vanilla-r4
# (path-int+RoPE) / PoPE-Flat (index+PoPE) / MapPoPE-Flat-r4 (the combination).
# n=1 again -- shape first at the corrected budget, seeds after if it is close.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
OUT="$REPO/enwik8_long"; mkdir -p "$OUT"
LOG="$REPO/enwik8_long.log"; echo "enwik8 long start $(date)" > "$LOG"
SEQ=512; BS=16; ITERS=36000; LR=2e-4
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
run(){ local NAME="$1" R="$2" TAG="$3"
  # guard on COMPLETION: the trainer used to write the final filename at every
  # eval, so a 1k-iter partial looked like a finished 36k run to a plain -f test.
  if [ -f "$OUT/${NAME}${TAG}.json" ] && python3 -c "import json,sys; sys.exit(0 if 'wall_total_s' in json.load(open(sys.argv[1])) else 1)" "$OUT/${NAME}${TAG}.json" 2>/dev/null; then
    echo "skip $NAME$TAG (complete)" >> "$LOG"; return; fi
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 20
  done
  echo "$(date +%H:%M) $NAME$TAG (r=$R) -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_hourglass_enwik8 --model "$NAME" --tag "$TAG" \
    --seq-len $SEQ --batch-size $BS --iters $ITERS --lr $LR --eval-every 1000 \
    --dim 512 --heads 8 --n-layers 9 --bottleneck-r "$R" \
    --out "$OUT" --device "cuda:$GPU" > "$OUT/${NAME}${TAG}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 5
}
run RoPE 2 ""
run PoPE-Flat 2 ""
run Vanilla 4 "_r4"
run MapPoPE-Flat 4 "_r4"
wait
python3 -u - "$OUT" >> "$REPO/ENWIK8_LONG.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
R=sys.argv[1]
rows=[("RoPE","","index","RoPE","baseline"),("PoPE-Flat","","index","PoPE",""),
      ("Vanilla","_r4","path-int","RoPE","r=4"),("MapPoPE-Flat","_r4","path-int","PoPE","**combination**, r=4")]
out=["# enwik8 — 36k iters, rank 4 (both 12k caveats fixed)","",
     "12k run was budget-limited (every arm still improving; remaining gain > the "
     "between-model spread) and used r=2 where MapFormer's own language run uses "
     "r=4. Both fixed. seq 512, bs 16, lr 2e-4, param-matched to 0.03%. **Lower is "
     "better.** n=1.","",
     "| model | position | encoding | val bpc | vs RoPE | 12k value | note |",
     "|---|---|---|---|---|---|---|"]
old={"RoPE":1.5221,"PoPE-Flat":1.5303,"Vanilla":1.5505,"MapPoPE-Flat":1.5193}
def val(n,t):
    p=os.path.join(R,f"{n}{t}.json")
    if not os.path.exists(p): return None,None
    c=json.load(open(p))["curve"]
    return c[-1]["val_bpc"], (c[-1]["val_bpc"]-c[-2]["val_bpc"]) if len(c)>1 else None
base,_=val("RoPE","")
for n,t,pos,enc,note in rows:
    v,slope=val(n,t)
    if v is None: out.append(f"| {n}{t} | {pos} | {enc} | — | — | {old.get(n,'')} | {note} |"); continue
    d="" if base is None or n=="RoPE" else f"{v-base:+.4f}"
    out.append(f"| {n}{t} | {pos} | {enc} | **{v:.4f}** | {d} | {old.get(n,'')} | {note} |")
out+=["", "Final-checkpoint slopes (negative = still improving; if these are still",
      "large the run is STILL budget-limited and the ordering is not settled):"]
for n,t,_,_,_ in rows:
    v,slope=val(n,t)
    if slope is not None: out.append(f"- {n}{t}: {slope:+.4f} per 1000 iters")
print("\n".join(out))
PYEOF
touch "$REPO/.enwik8_long_done"
echo "$(date) DONE" >> "$LOG"
