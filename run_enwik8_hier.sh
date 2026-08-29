#!/usr/bin/env bash
# What does HIERARCHY buy on text, on top of path integration / PoPE?
#
# NOT YET RUN AT THE CORRECT BUDGET. The only hierarchical arm ever put on enwik8 was
# MapPoPE-Hier at the OLD 12k budget with the broken random-sample eval (1.6345), and
# it was param-confounded. The enwik8 "hierarchy" result in the repo is of the PLAIN
# Hourglass scaffold, not hierarchical MapFormer -- and it shows hierarchy slightly
# WORSE on bpc (1.4844 vs flat10 1.4727) at -18.75% FLOPs, i.e. efficiency not quality.
#
# PRIMARY COMPARISON (clean): MapWM-Hier vs MapWM-FlatHG at dim 880 are
# PARAMETER-IDENTICAL (28,371,016 both at r=4) -- the same 3-block Hourglass scaffold
# differing ONLY in whether the middle block sees pooled (k=2) or full-resolution
# tokens. This isolates POOLING with nothing else moving. Same control that gave the
# compositional-task result (+0.130, n=8).
#
# SECONDARY (exploratory): MapPoPE-Hier and PoPE-Hier, for the fuller 2x2x2 picture.
# TWO CAVEATS ON THESE, both flagged rather than hidden:
#  1. RANK. MapPoPE-Hier does NOT accept bottleneck_r -- verified, r=2 and r=4 give
#     identical param counts (9,724,672), so **kwargs swallows it. The FLAT MapPoPE
#     arms ran at r=4. So hier-vs-flat for the PoPE family confounds rank. MapWM-Hier
#     DOES accept it (28,368,376 -> 28,371,016) so the primary pair is unaffected.
#  2. SCAFFOLD. The Hourglass ignores --n-layers and uses its own depth, so at dim 880
#     the hier arms are 28.37M vs the flat arms' 28.64M (-0.96%). Internal spread
#     among hier arms is 0.02%, so hier-vs-hier is clean; hier-vs-flat is not.
#
# n=1, shape first (project rule) -- seeds only if something clears the noise.
# Config matches the 36k flat run exactly: deterministic val, seq 512, bs 16, lr 2e-4.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
OUT="$REPO/enwik8_long"; mkdir -p "$OUT"
LOG="$REPO/enwik8_hier.log"; echo "enwik8 hier queued $(date)" > "$LOG"
while [ ! -f "$REPO/.g8_converge_done" ]; do sleep 120; done
echo "$(date +%H:%M) grid-8 finished; starting" >> "$LOG"
SEQ=512; BS=16; ITERS=36000; LR=2e-4; DIM=880; MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
run(){ local NAME="$1" R="$2" TAG="$3"
  local F="$OUT/${NAME}${TAG}.json"
  if [ -f "$F" ] && python3 -c "import json,sys; sys.exit(0 if 'wall_total_s' in json.load(open(sys.argv[1])) else 1)" "$F" 2>/dev/null; then
    echo "skip ${NAME}${TAG}" >> "$LOG"; return; fi
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 20
  done
  echo "$(date +%H:%M) ${NAME}${TAG} dim=$DIM r=$R -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_hourglass_enwik8 --model "$NAME" --tag "$TAG" \
    --seq-len $SEQ --batch-size $BS --iters $ITERS --lr $LR --eval-every 1000 \
    --dim $DIM --heads 8 --n-layers 9 --bottleneck-r "$R" \
    --out "$OUT" --device "cuda:$GPU" > "$OUT/${NAME}${TAG}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 5
}
run MapWM-Hier    4 "_h880"     # primary pair: pooling ON
run MapWM-FlatHG  4 "_h880"     # primary pair: pooling OFF, param-identical
run MapPoPE-Hier  4 "_h880"     # exploratory (rank not applied -- see header)
run PoPE-Hier     4 "_h880"     # exploratory
wait
python3 -u - "$OUT" >> "$REPO/ENWIK8_HIER.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
import numpy as np
R=sys.argv[1]
def m5(f):
    p=os.path.join(R,f)
    if not os.path.exists(p): return None,None
    d=json.load(open(p)); c=d["curve"]; v=[x["val_bpc"] for x in c[-5:]]
    return float(np.mean(v)), d["params"]
out=["# enwik8 — what does HIERARCHY buy on top of path integration / PoPE?","",
     "36k iters, deterministic val, seq 512, dim 880 for all hierarchical arms. "
     "Mean of last 5 checkpoints. **Lower is better.** n=1 (shape first).","",
     "## Primary: pooling isolated (parameter-IDENTICAL pair)","",
     "| model | pooling | params | val bpc |","|---|---|---|---|"]
a,pa=m5("MapWM-Hier_h880.json"); b,pb=m5("MapWM-FlatHG_h880.json")
for nm,v,p,pool in (("MapWM-Hier",a,pa,"ON (k=2)"),("MapWM-FlatHG",b,pb,"OFF")):
    out.append(f"| {nm} | {pool} | {p:,} | {v:.4f} |" if v else f"| {nm} | {pool} | — | missing |")
if a and b:
    out+=["", f"**Effect of pooling: {a-b:+.4f}** (negative = hierarchy helps).",
          "Same isolation that gave +0.130 on the compositional task (n=8).",""]
out+=["## Exploratory: the other hierarchical arms","",
      "| model | params | val bpc |","|---|---|---|"]
for nm,f in (("MapPoPE-Hier","MapPoPE-Hier_h880.json"),("PoPE-Hier","PoPE-Hier_h880.json")):
    v,p=m5(f); out.append(f"| {nm} | {p:,} | {v:.4f} |" if v else f"| {nm} | — | missing |")
out+=["","### Flat arms at 36k for reference (dim 512, 28.6M, NOT param-matched to the",
      "### hier arms -- they are 0.96% larger, and the PoPE hier arms also differ in rank)","",
      "| model | val bpc |","|---|---|",
      "| RoPE | 1.3864 |","| PoPE-Flat | 1.3806 |","| Vanilla_r4 | 1.3841 |",
      "| MapPoPE-Flat_r4 | 1.3786 |","",
      "> Read the PRIMARY pair for the hierarchy question -- it is the only fully",
      "> controlled comparison here. Hier-vs-flat carries a -0.96% param gap, and for",
      "> the PoPE family also a rank gap (MapPoPE-Hier cannot accept bottleneck_r).",
      "> Prior expectation from enwik8: the plain Hourglass scaffold was slightly WORSE",
      "> on bpc (1.4844 vs 1.4727) at -18.75% FLOPs -- efficiency, not quality."]
print("\n".join(out))
PYEOF
touch "$REPO/.enwik8_hier_done"
echo "$(date) DONE" >> "$LOG"
