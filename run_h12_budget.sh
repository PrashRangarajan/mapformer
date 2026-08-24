#!/usr/bin/env bash
# Is the partial recovery at H=12 a property of finer quantisation, or just
# undertraining?
#
# CONTINUOUS_ALLOC.md: allocentric recoding restores the position effect
# COMPLETELY at 4 headings (+0.050 -> +0.488, n=8) but only PARTIALLY at
# Habitat's 12 (+0.110 -> +0.263). Two live explanations:
#
#   (a) the fix genuinely degrades when displacement is finely quantised and
#       position is real-valued
#   (b) it is fine, and H=12 is undertrained -- the scored rate is 0.022 against
#       the torus baseline's 0.225, and Vanilla reached only 0.772 against a
#       0.509 floor, nowhere near the 0.996 it hits at H=4
#
# This is rule 5. The same check turned `rotate` from +0.004 (both arms sitting
# on the floor, a false negative) into +0.050. It also decides whether a Habitat
# port has a clean headline prediction, so it is worth an hour before committing
# days of engineering.
#
# Sweep the budget; 980 already exists and is reused.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/h12_budget; mkdir -p "$R"
for NB in 2000 4000; do
  for SEED in 0 1 2; do
    for V in Vanilla RoPE; do
      D="$R/nb${NB}/${V}_s${SEED}"
      [ -f "$D/${V}.pt" ] && { echo "skip nb$NB $V s$SEED"; continue; }
      python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
        --epochs 16 --n-batches "$NB" --batch-size 128 --n-steps 128 \
        --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
        --grid-size 64 --n-obs-types 16 --action-mode rotate --score-moves-only \
        --n-headings 12 --action-record allocentric \
        --device cuda:1 --output-dir "$D" \
        > "$R/train_nb${NB}_${V}_s${SEED}.log" 2>&1 &
    done
    wait
  done
  N=$(find "$R/nb$NB" -name '*.pt' 2>/dev/null | wc -l)
  [ "$N" -lt 6 ] && { echo "FAILED nb$NB: only $N/6 -- see $R/train_nb${NB}_*.log"; exit 1; }
  echo "$(date +%H:%M) nb=$NB done ($N/6)"
done
echo TRAINED; touch "$R/.trained"
