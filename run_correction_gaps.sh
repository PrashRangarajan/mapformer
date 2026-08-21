#!/usr/bin/env bash
# Two jobs of very different scientific value, run together.
#
# JOB 1 (integrity): gate lm200. LM200_CORRECTED_MULTISEED.md is cited in
# BASELINE_TABLE.md and has never had a context-destruction ablation (rule 2) --
# in the one regime where an entire leaderboard already turned out to be an
# artifact. Note what lm200 IS, though: the landmark extension was built as "the
# regime where Kalman/PC corrections have sharp measurements"
# (environment.py's own docstring). It is not the paper's task, and the
# correction was designed for it, so a win there is close to circular. This
# gates a cited number; it does not produce a result to build on.
#
# JOB 2 (the actual science): put the correction on tasks it was NOT designed
# for. Level15 has never been run on compositional or the family tree. Together
# with Match-Query (where it showed no advantage) these are the only
# non-circular tests of whether the correction generalises past its home turf.
# The family-tree arm also fills a second gap -- that task has no plain-WM arm at
# all, so its WM/EM axis is currently untested.
#
# Every comparison is within-batch (rule 3): Vanilla is retrained alongside
# Level15 everywhere, never taken from an existing directory.
# No `local` anywhere: under `set -u` it expands every word before assigning.
set -euo pipefail
cd "$(dirname "$0")/.."
R=mapformer/runs/correction_gaps
mkdir -p "$R"

# GPU dispatch, weighted 2:1 toward GPU 1.
#
# The previous rule was `if [ $((I % 2)) -eq 0 ]; then G=0; else G=1; fi` with I
# resetting every round. For a round of THREE variants that assigns gpu0, gpu1,
# gpu0 -- two on GPU 0 every single round, systematically. And GPU 0 is the one
# another user (mmattb) is saturating at 100% with a 12.8 GB job, so the bias
# landed our extra work on the contended device while GPU 1 idled at 11%.
#
# GLOBAL does not reset between rounds, so odd-sized rounds alternate their
# starting device instead of always beginning on gpu0. The 1 1 0 cycle then puts
# two thirds of the work on the free GPU.
GPUQ="1 1 0"
GLOBAL=0

echo "=== JOB 2a: family tree (fills the correction gap AND the plain-WM gap) ==="
for SEED in 0 1 2; do
  :
  for V in Vanilla Level15; do
    D="$R/familytree/seed${SEED}"
    [ -f "$D/${V}_familytree.pt" ] && { echo "skip ft $V s$SEED"; continue; }
    G=$(echo $GPUQ | cut -d' ' -f$(( GLOBAL % 3 + 1 )))
    GLOBAL=$((GLOBAL + 1))
    python3 -u -m mapformer.train_family_tree --variant "$V" --seed "$SEED" \
      --epochs 100 --n-batches 48 --batch-size 16 --depth 5 --n-obs 8 \
      --n-steps 64 --eval-steps 64 128 --n-layers 2 \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/ft_${V}_s${SEED}.log" 2>&1 &
  done
  wait
done
echo "$(date +%H:%M) family tree done"

echo "=== JOB 1: lm200 retrain (no checkpoints exist on this machine) ==="
for SEED in 0 1 2; do
  :
  for V in Vanilla Level15 Vanilla_ExtraHead; do
    D="$R/lm200/${V}_s${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip lm200 $V s$SEED"; continue; }
    G=$(echo $GPUQ | cut -d' ' -f$(( GLOBAL % 3 + 1 )))
    GLOBAL=$((GLOBAL + 1))
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --n-landmarks 200 --grid-size 64 --epochs 50 --n-batches 156 \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/lm200_${V}_s${SEED}.log" 2>&1 &
  done
  wait
done
echo "$(date +%H:%M) lm200 training done"

echo "=== JOB 1b: the ablation lm200 has never had ==="
python3 -u -m mapformer.ablate_paper_task --runs-dir "$R/lm200" \
  --variants Vanilla Level15 Vanilla_ExtraHead --seeds 0 1 2 \
  --n-batches 12 --batch-size 64 --n-steps 128 --device cuda:1 \
  --out mapformer/LM200_ABLATION.md > "$R/ablate.log" 2>&1
echo "$(date +%H:%M) lm200 ablation done"

echo "=== JOB 2b: compositional ==="
for SEED in 0 1 2; do
  :
  for V in Vanilla Level15; do
    D="$R/compositional/seed${SEED}"
    [ -f "$D/${V}.pt" ] && { echo "skip comp $V s$SEED"; continue; }
    G=$(echo $GPUQ | cut -d' ' -f$(( GLOBAL % 3 + 1 )))
    GLOBAL=$((GLOBAL + 1))
    python3 -u -m mapformer.train_compositional --variant "$V" --target motif \
      --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 \
      --n-templates 4 --seed "$SEED" \
      --device "cuda:$G" --output-dir "$D" \
      > "$R/comp_${V}_s${SEED}.log" 2>&1 &
  done
  wait
done
python3 -u -m mapformer.eval_compositional \
  --checkpoints $(ls "$R"/compositional/seed*/*.pt) \
  --lengths 256 1024 --n-traj 128 --batch 32 --device cuda:1 \
  --out mapformer/CORRECTION_COMPOSITIONAL.md > "$R/eval_comp.log" 2>&1
echo DONE; touch "$R/.done"
