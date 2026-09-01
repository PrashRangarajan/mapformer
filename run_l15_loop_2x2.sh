#!/usr/bin/env bash
# ARE THE FILTER AND THE LOOP COMPLEMENTARY? -- the 2x2 that was never run.
#
# THE SETUP. On the clean torus task the two mechanisms have exactly
# anti-correlated profiles, and the SAME axis carries both:
#
#             T=128    T=512    T=1024
#   Vanilla   0.966    0.891    0.768
#   Level15   0.979    0.948    0.888     <- benefit only at OOD length
#   Looped    1.000    0.816    0.642     <- benefit only at training length
#
# The loop's entire cost is OOD length; the filter's ONLY established effect is
# OOD length (+0.062 t 3.08 at T=512, +0.124 t 3.83 at T=1024, loss-matched,
# L15_ABLATION.md). Each one's win is the other's loss. No arm with BOTH has ever
# existed, so this has never been tested.
#
# TWO REASONS IT MIGHT NOT WORK, stated before running so the null is readable:
#   1. The filter does not obviously target the loop's failure mode. The loop's OOD
#      damage was MEASURED to be iteration count (same weights: T=512 peaks at 2
#      passes, falls monotonically to 6) and explicitly NOT residual growth (norm
#      flat, 18.15 -> 18.71). The filter bounds theta. Bounding theta has no evident
#      purchase on an iteration-count problem.
#   2. A free fix already exists. LoopedSampled repairs most of the collapse
#      (0.816 -> 0.915 at T=512) with no filter and no parameters. It is included
#      here as a 5th arm so the comparison is WITHIN-BATCH (rule 3): the filter has
#      to beat sampling, not merely beat nothing.
#
# PRIMARY MEASURE: the per-seed interaction
#     I = (Level15Looped - Looped) - (Level15 - Vanilla)
# at T=512 and T=1024. Parameter-matched by construction: the loop adds 0 params on
# both rows and the filter adds exactly 49,600 on both, so only the FILTER MAIN
# EFFECT carries a capacity difference -- the interaction does not.
#
# PRE-REGISTERED:
#   I > +MDE  -> COMPLEMENTARY (super-additive): the filter buys MORE inside the
#                loop than it does alone; the loop's OOD cost is what it repairs.
#   |I| < MDE -> NOT super-additive. Then read the LEVELS: if Level15Looped >=
#                max(Level15, Looped) at every length the two are merely ADDITIVE
#                and the combination is still the best arm; if not, they are not
#                complementary in any useful sense.
#   I < -MDE  -> they INTERFERE.
#   Separately, Level15Looped vs LoopedSampled decides whether the filter is worth
#   49,600 parameters over the free fix. A filter that only matches sampling is not
#   the answer to the loop's length problem.
#
# ANALYSIS PLAN, fixed in advance. Rule 9 applies: r(final loss, accuracy) was
# -0.930/-0.897/-0.812 over the ablation's 30 runs, and loss-matching FLIPPED two
# readings there in opposite directions. So the loss-matched residual contrast is
# PRIMARY here and the raw contrast is reported beside it. Both go in the file
# whichever way they point.
#
# POWER. n=12. The ablation's paired sd for Level15-Vanilla was 0.119 at T=512;
# an interaction roughly doubles that, so raw MDE at n=12 is ~0.14 -- marginal
# against an expected effect near +0.126 if the filter fully rescues the loop.
# Loss-matching roughly halved the effective noise there (t 1.86 -> 3.83), which is
# what makes n=12 adequate. If the loss-matched contrast also lands inside its MDE,
# the honest verdict is UNMEASURED, not null (rule 11).
#
# Clean config only. Noise was considered and REJECTED as the primary condition:
# at p=0.25 every arm sits within 0.11 of the 0.500 blank floor at T=512
# (0.569/0.593/0.606), so an OOD interaction would be floor-compressed exactly
# where it needs to be measured.
#
# 5 arms x 12 seeds = 60 runs, ~45 min each at 8-way concurrency -> ~5.6 h.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/l15_loop_2x2"; mkdir -p "$R/p0"
LOG="$REPO/l15_loop_2x2.log"; echo "l15-loop 2x2 start $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"; MAXPG=4
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"
  OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --n-steps $T --n-layers 1 \
    --n-heads $NH --d-model $DM --n-landmarks 0 --schedule cosine \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
  sleep 8
}
# arm varies fastest so a partial run still spans every arm rather than every seed
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  for V in Vanilla Level15 Looped Level15Looped LoopedSampled; do launch "$V" "$SEED"; done
done
wait
N=$(find "$R" -name '*.pt' | wc -l); echo "$(date +%H:%M) $N/60 checkpoints" >> "$LOG"

python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Level15 Looped Level15Looped LoopedSampled \
  --noises 0.0 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --lengths 128 512 1024 \
  --n-trials 100 --device cuda:0 --out "$REPO/_L15_LOOP_RAW.md" >> "$LOG" 2>&1

if [ -f "$REPO/_L15_LOOP_RAW.json" ]; then
  echo "$(date +%H:%M) eval OK; aggregating" >> "$LOG"
  python3 -u -m mapformer.agg_l15_loop --repo "$REPO" \
    --json "$REPO/_L15_LOOP_RAW.json" --runs-dir "$R" \
    --out "$REPO/L15_LOOP_2X2.md" >> "$LOG" 2>&1
else
  echo "$(date +%H:%M) EVAL FAILED -- no json" >> "$LOG"
fi
touch "$REPO/.l15_loop_2x2_done"; echo "$(date) DONE" >> "$LOG"
