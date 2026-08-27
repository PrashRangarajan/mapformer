#!/usr/bin/env bash
# CAPACITY CONTROL for GateDelta. GateDelta beat Vanilla by +0.082 (T=512) and
# +0.225 (T=1024) among pairs where BOTH arms converged -- the only effect this
# week to GROW under convergence-conditioning rather than vanish. But it carries
# +32,896 params (+1.0%). GateDeltaCtl has the SAME parameters (3,206,682,
# verified identical), created in the same order so the init RNG shift matches,
# evaluated in the forward pass -- but the gate output is multiplied out, so the
# model is functionally IDENTICAL to Vanilla (verified max|diff| = 0.00e+00) and
# the gate receives ZERO gradient (verified).
#
# DECISION RULE. If GateDeltaCtl ~= GateDelta, the win is capacity/RNG and the
# finding dies (this is exactly how Vanilla_ExtraHead killed the Level-1.5
# accuracy claim at t=0.79). If GateDeltaCtl ~= Vanilla, the win is the GATE --
# a component from the LANGUAGE paper improving the NAVIGATION model.
# Grids 16/24/32 x seeds 0,1,2 = 9 arms (~2.5h), matching GateDelta exactly.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mw_grid_sweep"; mkdir -p "$R"
LOG="$REPO/miniworld_gatectl.log"; echo "gate-control start $(date)" > "$LOG"
T=512; NBUF=24000; EP=100; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
for G in 16 24 32; do
  for SEED in 0 1 2; do
    OUT="$R/g${G}/s${SEED}"
    [ -f "$OUT/GateDeltaCtl_oracle.pt" ] && { echo "skip g${G}_s${SEED}" >> "$LOG"; continue; }
    while :; do
      PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
      if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
      sleep 15
    done
    echo "$(date +%H:%M) g${G}_GateDeltaCtl_s${SEED} -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_miniworld --variant GateDeltaCtl --seed "$SEED" --oracle \
      --grid-size $G --n-steps $T --buffer-size $NBUF --epochs $EP --n-batches $NB \
      --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH --n-workers $NW \
      --eval-trials $ETRIALS --eval-lengths 512 1024 --device "cuda:$GPU" \
      --output-dir "$OUT" > "$R/g${G}_GateDeltaCtl_s${SEED}.log" 2>&1 &
    PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
  done
done
wait
python3 -u - "$R" >> "$REPO/MINIWORLD_GATE_CONTROL.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np, torch
R = sys.argv[1]; TH = 0.4
def cell(g,s,v,L):
    j=os.path.join(R,f"g{g}",f"s{s}",f"{v}_oracle.json"); p=j.replace('.json','.pt')
    if not (os.path.exists(j) and os.path.exists(p)): return None,None
    return json.load(open(j))[str(L)]["nb_acc"], torch.load(p,map_location="cpu")["losses"][-1]
out=["# GateDelta capacity control","",
     "GateDeltaCtl = GateDelta's parameters (3,206,682, identical) with the gate "
     "multiplied out: functionally identical to Vanilla (max|diff| 0.00e+00), gate "
     "gradient exactly zero. If Ctl ~= GateDelta the win is capacity/RNG; if "
     "Ctl ~= Vanilla the win is the GATE.",""]
for L in (512,1024):
    out+=[f"## T={L}","","| grid | seed | Vanilla | GateDeltaCtl | GateDelta | Gate−Ctl | all converged? |","|---|---|---|---|---|---|---|"]
    gc, gc_conv = [], []
    for G in (16,24,32):
        for s in (0,1,2):
            va,vl=cell(G,s,"Vanilla",L); ca,cl=cell(G,s,"GateDeltaCtl",L); ga,gl=cell(G,s,"GateDelta",L)
            if None in (va,ca,ga): out.append(f"| {G} | {s} | — | — | — | — | missing |"); continue
            d=ga-ca; gc.append(d); ok=(vl<TH and cl<TH and gl<TH)
            if ok: gc_conv.append(d)
            out.append(f"| {G} | {s} | {va:.3f} <sub>({vl:.2f})</sub> | {ca:.3f} <sub>({cl:.2f})</sub> | "
                       f"{ga:.3f} <sub>({gl:.2f})</sub> | {d:+.3f} | {'YES' if ok else 'no'} |")
    out+=["",f"- **GateDelta − Control, pooled** ({len(gc)}): **{np.mean(gc):+.3f}**" if gc else "- no pairs",
          f"- **GateDelta − Control, all-converged** ({len(gc_conv)}): "
          + (f"**{np.mean(gc_conv):+.3f}**" if gc_conv else "*none converged together*"),""]
print("\n".join(out))
PYEOF
touch "$REPO/.mw_gatectl_done"
echo "$(date) DONE" >> "$LOG"
