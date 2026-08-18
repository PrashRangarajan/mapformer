#!/usr/bin/env bash
# Does the CORRECTION line survive on the tasks the PATH-INTEGRATION line uses?
#
# The project's two surviving result lines have never met. Level 1.5's evidence
# (+24.8pp over Vanilla on lm200 OOD, LM200_CORRECTED_MULTISEED.md) is entirely
# from regimes WITH observations. Match-Query's evidence is entirely from a
# regime WITHOUT them -- its query phase feeds MASK back, so the InEKF's
# innovation z - theta_hat is computed from an uninformative token at every
# scored step and the filter should collapse to plain path integration.
#
# Either outcome is informative:
#   Level15 == Vanilla on Match-Query -> the correction is measurement-driven,
#              which cleanly BOUNDS where it applies.
#   Level15 >  Vanilla on Match-Query -> it is not inference at all, which is the
#              sharpest possible demonstration of the "stabilisation, not
#              inference" reframing in CLAUDE.md.
#
# THREE ARMS, ALL TRAINED IN THE SAME BATCH (standing rule 3: never compare a
# fresh variant against a stored baseline). Vanilla is retrained here even though
# checkpoints exist, so the comparison is within-batch.
#
#   Vanilla            204,630 / 601,174 params (L=1 / L=3)
#   Vanilla_ExtraHead  270,934 / 667,478   <- MORE params than Level15
#   Level15            254,230 / 650,774
#
# Vanilla_ExtraHead is the capacity control and it is deliberately conservative:
# it has more parameters than Level15, so if Level15 wins it is not capacity.
# This control has form -- EXTRAHEAD_CONTROL.md used it to overturn the Hopfield
# structural claim ("CAPACITY, not structure").
#
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/level15_meets
mkdir -p "$R"
VARS="Vanilla Vanilla_ExtraHead Level15"

echo "=== PHASE 1: paper task (1 layer, paper config) ==="
for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/paper/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
      --device cuda:1 --output-dir "$D" \
      > "$R/paper_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "  paper seed $SEED done"
done

echo "=== PHASE 2: Match-Query (3 layers, TE=512 TQ=256, 200 epochs) ==="
for SEED in 0 1 2; do
  for V in $VARS; do
    D="$R/matchq/seed${SEED}"
    [ -f "$D/${V}_matchquery.pt" ] && { echo "skip $V s$SEED"; continue; }
    python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
      --epochs 200 --n-batches 48 --batch-size 16 \
      --T-explore 512 --T-query 256 --eval-query 256 512 \
      --n-layers 3 --d-model 128 --n-heads 2 \
      --device cuda:1 --output-dir "$D" \
      > "$R/matchq_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "  match-query seed $SEED done"
done

echo "=== PHASE 3: evaluate ==="
python3 -u -m mapformer.eval_paper_task --runs-dir "$R/paper" \
  --variants Vanilla Vanilla_ExtraHead Level15 --seeds 0 1 2 \
  --device cuda:1 --out mapformer/LEVEL15_MEETS_GATED_paper.md \
  > "$R/eval_paper.log" 2>&1
python3 - <<'PY' > mapformer/LEVEL15_MEETS_GATED_matchq.md
import json, glob, numpy as np, pathlib
R = pathlib.Path("mapformer/runs/level15_meets/matchq")
V = ["Vanilla", "Vanilla_ExtraHead", "Level15"]
out = ["# Level 1.5 on Match-Query -- does the correction survive with no measurements?", "",
       "Match-Query's query phase feeds MASK back, so the InEKF innovation "
       "`z - theta_hat` is computed from an uninformative token at every scored "
       "step. If the correction is measurement-driven it should collapse to plain "
       "path integration here.", "",
       "All three arms trained in the SAME batch (standing rule 3). "
       "`Vanilla_ExtraHead` has MORE parameters than Level15 (667,478 vs 650,774) "
       "and is the capacity control. **Chance 0.0625; never-moved floor 0.0893.**", "",
       "| variant | params | T_query=256 (train) | T_query=512 (OOD) |", "|---|---|---|---|"]
P = {"Vanilla": "601,174", "Vanilla_ExtraHead": "667,478", "Level15": "650,774"}
per = {}
for v in V:
    rows = {}
    for f in sorted(glob.glob(str(R / "seed*" / f"{v}_matchquery.json"))):
        d = json.load(open(f))
        for k, val in d.items():
            rows.setdefault(k, []).append(val["match_acc"])
    if not rows: continue
    per[v] = rows
    cells = []
    for k in ("256", "512"):
        a = np.array(rows.get(k, []))
        cells.append(f"{a.mean():.4f} ± {a.std(ddof=1):.4f}" if len(a) > 1 else
                     (f"{a.mean():.4f} (n=1)" if len(a) else "—"))
    out.append(f"| {v} | {P[v]} | " + " | ".join(cells) + " |")
out += ["", "## Per-seed (T_query=256)", ""]
for v, rows in per.items():
    out.append(f"- `{v}`: " + ", ".join(f"{x:.4f}" for x in rows.get("256", [])))
print("\n".join(out))
PY
echo DONE; touch "$R/.done"
