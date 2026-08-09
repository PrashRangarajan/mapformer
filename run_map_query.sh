#!/usr/bin/env bash
# Map-Query task: 6 variants x 3 seeds, T_explore=256 (the minimum operating
# point the gates validated; T=64 FAILS assume-start at 0.623).
# Gates are re-run as part of the pipeline so they ship next to the headline.
# NB: inside `local`, reference $1/$2 -- never $s. Bash expands all words of a
# `local` command BEFORE assigning, so `local s=$2 o=${s}` dies under `set -u`,
# silently if stderr is discarded.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/map_query.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"

echo "=== re-running pre-flight gates ===" >> "$LOG"
python3 -u -m mapformer.validate_map_query --n-episodes 1500 \
  --out "$REPO/MAP_QUERY_GATES.md" >> "$LOG" 2>&1

OUT="$REPO/runs/map_query"
VS=(Vanilla Hourglass_k2 PlainFlat PlainHourglass PoPE MapPoPE_Hier)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$OUT/seed$2"
  o="$OUT/seed$s"
  [ -f "$o/${v}_mapquery.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_map_query --variant "$v" --seed "$s" \
    --epochs 25 --n-batches 64 --batch-size 32 --T-explore 256 \
    --n-queries 8 --n-room-queries 4 --eval-explore 256 512 1024 \
    --n-layers 3 --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done ) & P1=$!
wait $P0 $P1

python3 -u - "$OUT" "$REPO/MAP_QUERY_RESULTS.md" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
out, dst = Path(sys.argv[1]), Path(sys.argv[2])
DISP={"Vanilla":"MapWM-Flat","Hourglass_k2":"MapWM-Hier","PlainFlat":"Plain-Flat",
      "PlainHourglass":"Plain-Hier","PoPE":"PoPE-Flat","MapPoPE_Hier":"MapPoPE-Hier"}
VS=list(DISP); LS=["256","512","1024"]
res={v:{L:{"direction_acc":[],"room_acc":[]} for L in LS} for v in VS}
for v in VS:
    for s in (0,1,2):
        f=out/f"seed{s}"/f"{v}_mapquery.json"
        if not f.exists(): print("MISSING",f); continue
        d=json.load(open(f))
        for L in LS:
            if L in d:
                for k in ("direction_acc","room_acc"): res[v][L][k].append(d[L][k])
def c(xs):
    if not xs: return "n/a"
    return f"{st.mean(xs):.3f}"+(f" ± {st.stdev(xs):.3f}" if len(xs)>1 else "")
ln=["# Map-Query results (n=3 seeds)","",
    "Trained at T_explore=256 -- the minimum operating point the gates validate.",
    "Held-out env (seed=10000). Gates in `MAP_QUERY_GATES.md`.","",
    "**Chance: direction ~0.50, room 0.016.** Oracle 1.000 for both.","",
    "## Room identity (the cognitive-map metric, 64 classes)","",
    "| variant | "+" | ".join(f"T={L}" for L in LS)+" |","|---"*(len(LS)+1)+"|"]
for v in VS: ln.append(f"| {DISP[v]} | "+" | ".join(c(res[v][L]["room_acc"]) for L in LS)+" |")
ln+=["","## Goal direction (chance ~0.50)","",
     "| variant | "+" | ".join(f"T={L}" for L in LS)+" |","|---"*(len(LS)+1)+"|"]
for v in VS: ln.append(f"| {DISP[v]} | "+" | ".join(c(res[v][L]["direction_acc"]) for L in LS)+" |")
dst.write_text("\n".join(ln)+"\n"); print("\n".join(ln))
PY

cd "$REPO"
git add MAP_QUERY_RESULTS.md MAP_QUERY_GATES.md MAP_QUERY_GATES.json \
  environment_map_query.py train_map_query.py run_map_query.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Map-Query results: 6 variants x 3 seeds at T_explore=256

Task scores ONE token per query instead of a sequence, so an n-gram over answers
is at chance by construction. Gates re-run in-pipeline and shipped alongside.
Chance: direction ~0.50, room 0.016; oracle 1.000 both.
Auto-committed; interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.map_query_done"
