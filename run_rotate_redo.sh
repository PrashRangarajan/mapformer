#!/usr/bin/env bash
# Rotate condition, redone. The first attempt was VOID: revisit was keyed on
# (x, y, heading) while the allo observation depends only on (x, y), and turns
# do not translate, so a run of "turn left" emitted the same observation
# repeatedly -- 93% solvable by copying the previous answer (KNOB_SWEEP.md).
#
# TWO fixes, both verified by gating BEFORE training this time (rule 1, which I
# skipped last round):
#   1. revisit keyed on the OBSERVATION-DETERMINING state, not (x,y,heading)
#   2. --score-moves-only: skip steps where the observed cell did not change
# Gate after: o1..o5 = 0.501/0.472/0.440/0.462 against a 0.507 marginal. PASS.
#
# TWO BUDGETS, because the fix costs supervision. The scored rate falls from
# 0.727 to 0.054 (baseline is 0.225), so at equal epochs rotate gets ~4x fewer
# gradient-contributing events than baseline and a weak effect could be
# undertraining rather than the knob (rule 5). "matched" gives it 4x the batches
# so the number of scored events is comparable; "standard" keeps the sweep's
# recipe. Both reported.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/knob_sweep
for CFG in "standard:98" "matched:392"; do
  LBL="${CFG%%:*}"; NB="${CFG#*:}"
  for SEED in 0 1 2; do
    for V in Vanilla RoPE; do
      D="$R/rotate_${LBL}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && { echo "skip $LBL $V s$SEED"; continue; }
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches "$NB" --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        --grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only \
        --device cuda:1 --output-dir "$D" \
        > "$R/train_rotate_${LBL}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  echo "$(date +%H:%M) rotate_$LBL done"
done
echo DONE; touch "$R/.rotate_done"
