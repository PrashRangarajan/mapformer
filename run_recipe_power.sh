#!/usr/bin/env bash
# STEP 1: FIX POWER BEFORE CHANGING THE TASK.
#
# THE PROBLEM, measured on this session's own 60 runs. Effects worth chasing here
# are 0.02-0.10. Minimum detectable effects are not:
#
#   arm        sd @ T=1024   MDE @ n=12
#   Level15       0.092         0.074
#   Vanilla       0.084         0.068
#   Looped        0.166         0.134
#
# We are underpowered by 2-5x, which is why most results this session read
# "unmeasured" rather than "null". A harder task at the same variance buys nothing.
#
# The variance is NOT intrinsic -- it is bimodal training. Vanilla's final loss
# ranged 0.0087-0.6099 over 12 seeds and Level15's 0.0001-0.4198, under the CURRENT
# warmup+cosine recipe. A few seeds simply never leave the plateau. Rule 10 already
# established this landscape is plateau-then-cliff (LinearLR-from-step-one moved one
# arm 0.448 -> 0.990); warmup+cosine helped but did not finish the job.
#
# So the question is not "which architecture" but "which recipe converges 8/8".
# Halving sd is worth more than any new benchmark and costs one batch.
#
# CONDITIONS. Only knobs that already exist, no code change:
#   C0  300 ep, lr 3e-4   -- the current recipe, the control
#   C1  300 ep, lr 1e-3   -- reach the cliff sooner
#   C2  600 ep, lr 1e-3   -- more steps at a usable LR
# Two arms: Vanilla (the plain case) and Looped (the WORST variance, sd 0.166).
#
# PRE-REGISTERED. Primary metric is CONVERGED FRACTION (final loss < 0.05 and
# |slope| over the last 10% < 5e-4), secondary is sd of held-out accuracy.
#   a condition reaching 8/8 converged on BOTH arms -> adopt it for step 2
#   no condition beating C0's converged fraction    -> the bimodality is not the
#       LR schedule, and the next suspects are init scale and dropout, neither of
#       which is a CLI knob yet
# Mean accuracy is reported but is NOT the criterion: picking a recipe on mean
# accuracy over 8 seeds is how you select a lucky basin.
#
# NOTE ON THE DATA PATH. This is the first batch to use --data-workers (verified
# tonight: worker fidelity, worker-count invariance, distributional equivalence,
# 2.15x at 3 workers). It draws a DIFFERENT sample from the same generator than the
# serial path, so C0 will not reproduce tonight's serial numbers exactly -- but it
# is the same distribution, and every arm here uses the same path, so the
# within-batch comparison is clean (rule 3). C0's converged fraction against
# tonight's serial 12 Vanilla seeds is itself a check on the new path.
#
# 2 arms x 3 conditions x 8 seeds = 48 runs.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/recipe_power"; mkdir -p "$R"
LOG="$REPO/recipe_power.log"; echo "recipe power start $(date)" > "$LOG"
NB=98; BS=128; T=128; DM=128; NH=2; DW=3
A="train_var""iant"; MAXPG=5      # 10 jobs x (1 main + 3 data workers) ~= 32 cores
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"; EP="$3"; LR="$4"; C="$5"
  OUT="$R/$C/p0/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $C $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $C $V s$SEED (${EP}ep lr$LR) -> cuda:$GPU ($N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs "$EP" --lr "$LR" --n-batches $NB --batch-size $BS --n-steps $T \
    --n-layers 1 --n-heads $NH --d-model $DM --n-landmarks 0 --schedule cosine \
    --data-workers $DW --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${C}_${V}_s${SEED}.log" 2>&1 &
  sleep 8
}
for SEED in 0 1 2 3 4 5 6 7; do
  for V in Vanilla Looped; do
    launch "$V" "$SEED" 300 3e-4 C0
    launch "$V" "$SEED" 300 1e-3 C1
    launch "$V" "$SEED" 600 1e-3 C2
  done
done
wait
N=$(find "$R" -name '*.pt' | wc -l); echo "$(date +%H:%M) $N/48 checkpoints" >> "$LOG"
for C in C0 C1 C2; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$C" \
    --variants Vanilla Looped --noises 0.0 --seeds 0 1 2 3 4 5 6 7 \
    --lengths 128 512 1024 --n-trials 100 --device cuda:0 \
    --out "$REPO/_RECIPE_${C}.md" >> "$LOG" 2>&1
done
python3 -u -m mapformer.agg_recipe --repo "$REPO" --runs-dir "$R" \
  --out "$REPO/RECIPE_POWER.md" >> "$LOG" 2>&1
touch "$REPO/.recipe_power_done"; echo "$(date) DONE" >> "$LOG"
