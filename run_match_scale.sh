#!/usr/bin/env bash
# Match-Query scale-up. Every config gated first (gates in the pipeline, and the
# floors MOVE: n_obs=4 puts chance at 0.25, not 0.0625).
#   A  64^2  n_obs=16  seeds 3,4   -> extends the base result to n=5
#   B 128^2  n_obs=16  seeds 0,1,2 -> 4x the map
#   C  64^2  n_obs=4   seeds 0,1,2 -> heavy aliasing, CSCG-like
# Variants: Vanilla (path integration) vs PlainFlat (index) -- the axis under test.
# NB: reference $1/$2 inside `local`, never $s.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/match_scale.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"

for cfg in "64 16" "128 16" "64 4"; do set -- $cfg
  python3 -u -m mapformer.validate_match_query --size $1 --n-obs $2 \
    --t-explore 512 --t-query 256 --out "$REPO/MATCH_GATES_${1}_${2}.md" >> "$LOG" 2>&1
done

JOBS=()
for s in 3 4; do for v in Vanilla PlainFlat; do JOBS+=("$v:$s:64:16:base"); done; done
for s in 0 1 2; do for v in Vanilla PlainFlat; do JOBS+=("$v:$s:128:16:big"); done; done
for s in 0 1 2; do for v in Vanilla PlainFlat; do JOBS+=("$v:$s:64:4:alias"); done; done
run(){ local j=$1; IFS=: read -r v s sz no tag <<<"$j"
  local o="$REPO/runs/match_scale_${tag}/seed${s}"
  [ -f "$o/${v}_matchquery.pt" ] && return
  echo "$(date +%H:%M) $tag $v s$s size=$sz n_obs=$no" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$v" --seed "$s" \
    --size "$sz" --n-obs "$no" --epochs 200 --n-batches 48 --batch-size 16 \
    --T-explore 512 --T-query 256 --eval-query 256 512 1024 --n-layers 3 \
    --device cuda:0 --output-dir "$o" >> "$LOG" 2>&1; }
for j in "${JOBS[@]}"; do run "$j"; done

python3 -u -m mapformer.eval_match_longq --runs-dir "$REPO/runs/match_query" \
  --lengths 256 512 1024 2048 --device cuda:0 \
  --out "$REPO/MATCH_QUERY_LONGQ.md" >> "$LOG" 2>&1

python3 -u - "$REPO" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
R = Path(sys.argv[1]); LS = ["256", "512", "1024"]
def cell(xs):
    # ALWAYS print n per cell: columns can have different seed counts when not
    # every seed was evaluated at every length, and a table whose n varies by
    # column without saying so is misleading (this happened, see the CORRECTION
    # banner in MATCH_QUERY_SCALE.md).
    if not xs: return "n/a"
    return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "") + f" (n={len(xs)})"
ln = ["# Match-Query scale-up (n=3, n=5 on base)", "",
      "Vanilla = path integration; PlainFlat = index position. Gates re-run per config.",
      "**Chance is 0.0625 at n_obs=16 and 0.2500 at n_obs=4.**", ""]
for tag, seeds, ch, desc in (("base", (0,1,2,3,4), 0.0625, "64^2, n_obs=16"),
                             ("big", (0,1,2), 0.0625, "128^2, n_obs=16 -- 4x the map"),
                             ("alias", (0,1,2), 0.2500, "64^2, n_obs=4 -- heavy aliasing")):
    ln += [f"## {tag}: {desc}  (chance {ch:.4f})", "",
           "| variant | " + " | ".join(f"TQ={L}" for L in LS) + " |", "|---"*(len(LS)+1)+"|"]
    for v in ("Vanilla", "PlainFlat"):
        acc = {L: [] for L in LS}
        for s in seeds:
            for d in (R/f"runs/match_scale_{tag}"/f"seed{s}", R/"runs/match_query"/f"seed{s}"):
                f = d/f"{v}_matchquery.json"
                if f.exists():
                    j = json.load(open(f))
                    for L in LS:
                        if L in j: acc[L].append(j[L]["match_acc"])
                    break
        ln.append(f"| {v} | " + " | ".join(cell(acc[L]) for L in LS) + " |")
    ln.append("")
(R/"MATCH_QUERY_SCALE.md").write_text("\n".join(ln)+"\n"); print("\n".join(ln))
PY

cd "$REPO"; git add MATCH_QUERY_SCALE.md MATCH_QUERY_LONGQ.md MATCH_QUERY_LONGQ.json \
  MATCH_GATES_*.md MATCH_GATES_*.json eval_match_longq.py run_match_scale.sh \
  train_match_query.py validate_match_query.py 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Match-Query scale-up: 128^2 grid, n_obs=4 aliasing, n=5 base, blind phase to 2048

Gates re-run per config; floors move (chance 0.25 at n_obs=4). Auto-committed;
interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.match_scale_done"
