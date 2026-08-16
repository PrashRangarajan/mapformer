#!/usr/bin/env bash
# EM on the Match-Query task: does the single-p_0 ablation help where the task
# is position MATCHING and its shortcuts are gated?
# Three arms trained in ONE batch (the standing rule -- never compare a fresh
# variant to a stored baseline):
#   Vanilla        MapFormer-WM. Also a reproducibility check: the earlier sweep
#                  gave 0.888 +/- 0.140 under identical settings.
#   VanillaEM      paper-faithful separate q0/k0 (App. A.4).
#   VanillaEM_P0   single-p_0 ablation of the paper's stated-but-untested
#                  conjecture that the separation helps.
# NB: inside `local`, reference $1/$2 -- never $s.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/match_query_em.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
OUT="$REPO/runs/match_query_em"
VS=(Vanilla VanillaEM VanillaEM_P0)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local j=$1; IFS=: read -r v s <<<"$j"
  local o="$OUT/seed$s"
  [ -f "$o/${v}_matchquery.pt" ] && return
  echo "$(date +%H:%M) $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$v" --seed "$s" \
    --epochs 200 --n-batches 48 --batch-size 16 --T-explore 512 --T-query 256 \
    --eval-query 256 512 --n-layers 3 --device cuda:0 \
    --output-dir "$o" >> "$LOG" 2>&1; }
for j in "${JOBS[@]}"; do run "$j"; done   # single GPU: run all sequentially

python3 -u - "$OUT" "$REPO/MATCH_QUERY_EM.md" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
out, dst = Path(sys.argv[1]), Path(sys.argv[2])
VS = ["Vanilla", "VanillaEM", "VanillaEM_P0"]
LAB = {"Vanilla":"MapWM-Flat (WM)","VanillaEM":"MapEM separate q0/k0 (paper-faithful)",
       "VanillaEM_P0":"MapEM single p_0 (ablation)"}
res = {v: {L: [] for L in ("256","512")} for v in VS}
for v in VS:
    for s in (0,1,2):
        f = out/f"seed{s}"/f"{v}_matchquery.json"
        if not f.exists(): print("MISSING", f); continue
        d = json.load(open(f))
        for L in ("256","512"):
            if L in d: res[v][L].append(d[L]["match_acc"])
def c(xs):
    if not xs: return "n/a"
    return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs)>1 else "")
ln = ["# EM on Match-Query — does single-p_0 help where the task is MATCHING?","",
      "All three arms trained in ONE batch, TE=512 TQ=256, 200 epochs, n=3.",
      "Held-out env (seed=10000). **Chance 0.0625.** Gates: `MATCH_QUERY_GATES.md`.","",
      "Match-Query is the only task in this repo whose shortcuts are gated, and it",
      "tests position MATCHING — which is what A_P is for. So it is the most",
      "informative place to ask whether the q0/k0 parameterisation matters.","",
      "| variant | T_query=256 | T_query=512 (OOD) |","|---|---|---|"]
for v in VS: ln.append(f"| {LAB[v]} | {c(res[v]['256'])} | {c(res[v]['512'])} |")
ln += ["","## Per-seed (T_query=256)","","| variant | s0 | s1 | s2 |","|---|---|---|---|"]
for v in VS:
    xs = res[v]["256"]
    ln.append(f"| {LAB[v]} | " + " | ".join(f"{x:.3f}" for x in xs) + " |" if xs else f"| {LAB[v]} | n/a |")
ln += ["","Reference: the earlier 6-variant sweep gave Vanilla 0.888 ± 0.140 under",
       "identical settings. The Vanilla arm here is the reproducibility check."]
dst.write_text("\n".join(ln)+"\n"); print("\n".join(ln))
PY

cd "$REPO"; git add MATCH_QUERY_EM.md run_match_query_em.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "EM on Match-Query: single-p_0 vs paper-faithful separate q0/k0

Same-batch: Vanilla, VanillaEM, VanillaEM_P0, 3 seeds, TE=512 TQ=256, 200 epochs.
Vanilla doubles as a reproducibility check against the sweep's 0.888 +/- 0.140.
Auto-committed; interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.match_query_em_done"
