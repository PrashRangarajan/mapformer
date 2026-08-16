#!/usr/bin/env bash
# Family-tree sweep: does MapEM-NC handle a real relational hierarchy?
#   MapEM_NC_L / MapEM_NC_NL  the paper's non-commutative variants (B.2.2)
#   VanillaEM_P0              COMMUTATIVE control -- SO(2) translations. The
#                             paper predicts this cannot represent the structure.
#   PlainFlat                 index-position control (no path integration at all)
# Effective floor is the HUB baseline 0.163, not chance 0.125.
# NB: reference $1/$2 inside `local`, never $s.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/family_tree.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
python3 -u -m mapformer.validate_family_tree --depth 5 --n-obs 8 \
  --out "$REPO/FAMILY_TREE_GATES.md" >> "$LOG" 2>&1
OUT="$REPO/runs/family_tree"
JOBS=(); for s in 0 1 2; do for v in MapEM_NC_L MapEM_NC_NL VanillaEM_P0 PlainFlat; do
  JOBS+=("$v:$s"); done; done
run(){ local j=$1; IFS=: read -r v s <<<"$j"
  local o="$OUT/seed$s"
  [ -f "$o/${v}_familytree.pt" ] && return
  echo "$(date +%H:%M) $v s$s" >> "$LOG"
  python3 -u -m mapformer.train_family_tree --variant "$v" --seed "$s" \
    --epochs 100 --n-batches 48 --batch-size 16 --depth 5 --n-obs 8 \
    --n-steps 64 --eval-steps 64 128 --n-layers 2 --device cuda:0 \
    --output-dir "$o" >> "$LOG" 2>&1; }
for j in "${JOBS[@]}"; do run "$j"; done

python3 -u - "$OUT" "$REPO/FAMILY_TREE_RESULTS.md" >> "$LOG" 2>&1 <<'PY'
import json, sys, statistics as st
from pathlib import Path
out, dst = Path(sys.argv[1]), Path(sys.argv[2])
V = [("MapEM_NC_L","MapEM-NC-L (non-commutative, linear)"),
     ("MapEM_NC_NL","MapEM-NC-NL (non-commutative, MLP)"),
     ("VanillaEM_P0","MapEM single-p0 (COMMUTATIVE control)"),
     ("PlainFlat","Plain-Flat (index position, no PI)")]
LS = ["64","128"]
def cell(xs):
    if not xs: return "n/a"
    return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs)>1 else "")
ln = ["# Family-tree (non-commutative relational structure) — results", "",
      "The task MapFormer's appendix B.2.2 motivates and never runs: mother and",
      "father do not commute. 8 relational actions, scored at revisited nodes,",
      "n=3 seeds. Trained at 64 steps, also evaluated at 128 (OOD length).", "",
      "**chance 0.125; effective floor is the HUB baseline 0.163** — shallow nodes",
      "are revisited more, so answering with the most-visited node's observation",
      "already scores 0.163. Read every number against 0.163.", "",
      "| variant | n_steps=64 (train) | n_steps=128 (OOD) |", "|---|---|---|"]
for k, lab in V:
    acc = {L: [] for L in LS}
    for s in (0,1,2):
        f = out/f"seed{s}"/f"{k}_familytree.json"
        if f.exists():
            j = json.load(open(f))
            for L in LS:
                if L in j: acc[L].append(j[L])
    ln.append(f"| {lab} | " + " | ".join(cell(acc[L]) for L in LS) + " |")
ln += ["", "## Per seed (n_steps=64)", "", "| variant | s0 | s1 | s2 |", "|---|---|---|---|"]
for k, lab in V:
    r = []
    for s in (0,1,2):
        f = out/f"seed{s}"/f"{k}_familytree.json"
        r.append(f"{json.load(open(f))['64']:.3f}" if f.exists() else "n/a")
    ln.append(f"| {lab} | " + " | ".join(r) + " |")
dst.write_text("\n".join(ln)+"\n"); print("\n".join(ln))
PY
cd "$REPO"; git add FAMILY_TREE_RESULTS.md FAMILY_TREE_GATES.md FAMILY_TREE_GATES.json run_family_tree.sh 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Family-tree results: MapEM-NC on a real relational hierarchy

The paper's motivating example for non-commutative groups, run for the first
time. Floor is the hub baseline 0.163, not chance 0.125. Auto-committed;
interpretation pending."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.family_tree_done"
