#!/usr/bin/env bash
# Match-Query sweep: 6 variants x 3 seeds, TE=512 TQ=256, 200 epochs.
# Budget from the Map-Query diagnostic (25 was 8x too few) and from this task's
# own probe: Vanilla was still descending at epoch 60 (held-out 0.297 vs chance
# 0.0625). Gates re-run in-pipeline so they ship beside the headline.
# NB: inside `local`, reference $1/$2 -- never $s (bash expands all words of a
# `local` command before assigning; dies silently under `set -u`).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/match_query.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"

echo "=== re-running pre-flight gates ===" >> "$LOG"
python3 -u -m mapformer.validate_match_query --t-explore 512 --t-query 256 \
  --out "$REPO/MATCH_QUERY_GATES.md" >> "$LOG" 2>&1

OUT="$REPO/runs/match_query"
VS=(Vanilla Hourglass_k2 PlainFlat PlainHourglass PoPE MapPoPE_Hier)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$OUT/seed$s"
  [ -f "$o/${v}_matchquery.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$v" --seed "$s" \
    --epochs 200 --n-batches 48 --batch-size 16 --T-explore 512 --T-query 256 \
    --eval-query 256 512 --n-layers 3 --device "cuda:$g" \
    --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done ) & P1=$!
wait $P0 $P1

python3 -u - "$OUT" "$REPO/MATCH_QUERY_RESULTS.md" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
out, dst = Path(sys.argv[1]), Path(sys.argv[2])
DISP={"Vanilla":"MapWM-Flat","Hourglass_k2":"MapWM-Hier","PlainFlat":"Plain-Flat",
      "PlainHourglass":"Plain-Hier","PoPE":"PoPE-Flat","MapPoPE_Hier":"MapPoPE-Hier"}
VS=list(DISP); LS=["256","512"]
res={v:{L:{"match_acc":[],"match_nll":[]} for L in LS} for v in VS}
for v in VS:
    for s in (0,1,2):
        f=out/f"seed{s}"/f"{v}_matchquery.json"
        if not f.exists(): print("MISSING",f); continue
        d=json.load(open(f))
        for L in LS:
            if L in d:
                for k in ("match_acc","match_nll"): res[v][L][k].append(d[L][k])
def c(xs):
    if not xs: return "n/a"
    return f"{st.mean(xs):.3f}"+(f" ± {st.stdev(xs):.3f}" if len(xs)>1 else "")
ln=["# Match-Query results (n=3 seeds)","",
    "Blind continuation: explore with observations revealed, then continue with",
    "them withheld and predict the observation at each cell. Scored at cells",
    "visited during explore and non-blank; each cell scored once per episode.","",
    "Trained TE=512 TQ=256, 200 epochs. Held-out env (seed=10000).",
    "**Chance 0.0625.** Gates in `MATCH_QUERY_GATES.md` (all at chance).","",
    "## Match accuracy","",
    "| variant | T_query=256 (train) | T_query=512 (OOD) |","|---|---|---|"]
for v in VS: ln.append(f"| {DISP[v]} | "+" | ".join(c(res[v][L]["match_acc"]) for L in LS)+" |")
ln+=["","## Match NLL (lower better)","",
     "| variant | T_query=256 | T_query=512 |","|---|---|---|"]
for v in VS: ln.append(f"| {DISP[v]} | "+" | ".join(c(res[v][L]["match_nll"]) for L in LS)+" |")
dst.write_text("\n".join(ln)+"\n"); print("\n".join(ln))
PY

cd "$REPO"
git add MATCH_QUERY_RESULTS.md MATCH_QUERY_GATES.md MATCH_QUERY_GATES.json run_match_query.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Match-Query sweep: 6 variants x 3 seeds, TE=512 TQ=256, 200 epochs

Probe justified the launch: Vanilla reached held-out 0.297 (chance 0.0625) at 60
epochs with the loss still descending. Gates all at chance. Auto-committed;
interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.match_query_done"
