#!/usr/bin/env bash
# LANGUAGE: does the PoPE x path-integration combination compose on text?
#
# THE GAP. MapFormer (v4 sec 5.5) and PoPE (ICML 2026) were BOTH evaluated on
# OpenWebText at ctx 1024 against a RoPE baseline, by different authors, and
# NEITHER cites the other. Their gains over RoPE are nearly identical:
#   MapFormer  19.14 -> 18.79  (-0.35 ppl, 1.8%)   ~100B tokens
#   PoPE 124M  21.55 -> 21.33  (-0.22, 1.0%)       ~9B tokens
#   PoPE 253M  18.88 -> 18.55  (-0.33, 1.7%)
# And they are ORTHOGONAL: PoPE changes how finely position is ENCODED (double the
# frequencies; content-dependent MAGNITUDE, positional ANGLE), while MapFormer
# changes what position MEANS (content-dependent ANGLE from a cumsum). Nobody has
# combined them on language. On NAVIGATION we already showed they compose: PoPE
# alone sits at the floor (0.509), path-integration alone 0.967, both 0.994.
#
# THE 2x2, param-matched to 0.03%:
#   RoPE          28,634,880   index    + RoPE   <- baseline
#   Vanilla       28,636,672   path-int + RoPE
#   PoPE-Flat     28,639,488   index    + PoPE
#   MapPoPE-Flat  28,642,048   path-int + PoPE   <- the untested combination
# plus two extras:
#   NoPE          28,634,880   no rotation at all. The baseline this subfield
#                 omits -- SRoPE runs it everywhere and it SOMETIMES WINS on
#                 language, while on OUR navigation task it collapsed to chance.
#                 Direct cross-domain contrast.
#   MapPoPE-Hier  29,404,288 at dim=896 (Hourglass ignores --n-layers and uses its
#                 own 3-block scaffold; at dim=512 it would be only 9.7M). +2.7%
#                 params vs the flat arms -- EXPLORATORY, flag it.
#
# OMEGA INIT. MapFormer sets omega geometrically so the lowest frequency completes
# one cycle over the largest traversable distance (omega_min = 2pi/grid_size).
# Language has no grid, so grid_size = seq_len: the slowest rotation spans exactly
# one context window. Selective RoPE instead DERIVES its ladder from RFF theory --
# the untested alternative (LANGUAGE_LANDSCAPE.md).
#
# Config matches the existing flat9 enwik8 run (seq 512, bs 16, 12000 iters, lr
# 2e-4) so results are comparable to hourglass_enwik8/. Waits for the GateDelta
# capacity control to free the GPUs.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
OUT="$REPO/enwik8_2x2"; mkdir -p "$OUT"
LOG="$REPO/enwik8_2x2.log"; echo "enwik8 2x2 queued $(date)" > "$LOG"
while [ ! -f "$REPO/.mw_gatectl_done" ]; do sleep 60; done
echo "$(date +%H:%M) GPUs free; starting" >> "$LOG"

SEQ=512; BS=16; ITERS=12000; LR=2e-4
MAXPG=2
declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
run(){  # name dim
  local NAME="$1" DIM="$2"
  [ -f "$OUT/${NAME}.json" ] && { echo "skip $NAME" >> "$LOG"; return; }
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 20
  done
  echo "$(date +%H:%M) $NAME (dim=$DIM) -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_hourglass_enwik8 --model "$NAME" \
    --seq-len $SEQ --batch-size $BS --iters $ITERS --lr $LR --eval-every 500 \
    --dim "$DIM" --heads 8 --n-layers 9 --out "$OUT" --device "cuda:$GPU" \
    > "$OUT/${NAME}.log" 2>&1 &
  PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 5
}
run RoPE 512
run Vanilla 512
run PoPE-Flat 512
run MapPoPE-Flat 512
run NoPE 512
run MapPoPE-Hier 896
wait
echo "$(date +%H:%M) done; aggregating" >> "$LOG"

python3 -u - "$OUT" >> "$REPO/ENWIK8_2X2.md" 2>>"$LOG" <<'PYEOF'
import json, os, sys
R = sys.argv[1]
rows = [("RoPE","index","RoPE","baseline"), ("Vanilla","path-int","RoPE",""),
        ("PoPE-Flat","index","PoPE",""), ("MapPoPE-Flat","path-int","PoPE","**the combination**"),
        ("NoPE","none","none","null arm"), ("MapPoPE-Hier","path-int","PoPE +hier","dim=896, +2.7% params")]
def bpc(n):
    p = os.path.join(R, f"{n}.json")
    if not os.path.exists(p): return None, None
    d = json.load(open(p))
    return d["curve"][-1]["val_bpc"], d["params"]
out = ["# enwik8 — PoPE x path-integration, and does the combination compose?", "",
       "byte-level enwik8, seq 512, batch 16, 12k iters, lr 2e-4 (matching the "
       "existing flat9 run). Flat arms param-matched to 0.03%. **Lower bpc is "
       "better.** n=1 — this is a shape-finding run, not a claim.", "",
       "| model | position | encoding | params | val bpc | vs RoPE | note |",
       "|---|---|---|---|---|---|---|"]
base, _ = bpc("RoPE")
for n, pos, enc, note in rows:
    b, pr = bpc(n)
    if b is None: out.append(f"| {n} | {pos} | {enc} | — | — | — | {note} |"); continue
    d = "" if base is None or n == "RoPE" else f"{b-base:+.4f}"
    out.append(f"| {n} | {pos} | {enc} | {pr:,} | **{b:.4f}** | {d} | {note} |")
out += ["", "> On NAVIGATION these compose: PoPE alone is at the floor, path-int alone",
        "> 0.967, both 0.994. The question here is whether that holds on text, where",
        "> each is worth only ~1-2% alone. Also watch NoPE: it collapsed to chance on",
        "> navigation but is competitive on language in Selective RoPE's own tables."]
print("\n".join(out))
PYEOF
touch "$REPO/.enwik8_2x2_done"
echo "$(date) DONE" >> "$LOG"
