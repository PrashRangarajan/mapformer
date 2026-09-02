#!/usr/bin/env bash
# STEP 2: the filter x loop 2x2 on a task where the FILTER'S PREMISE ACTUALLY HOLDS.
#
# WHY THIS AND NOT ANOTHER TORUS RUN. Every correction result in this project has
# been measured where the correction had nothing to correct. Match-Query's actions
# are clean and its query phase is blind, so testing refine-theta there was a design
# error that burned 16 runs replicating a known negative (rule 17). The clean torus
# has drift but is at ceiling -- last night's 2x2 put every arm between 0.947 and
# 0.999 at training length, and its only discrimination came from OOD length where
# the seed sd is 0.12-0.24.
#
# Stochastic transitions on Match-Query fix both at once. Explore now RECORDS the
# commanded action and EXECUTES a resampled one with probability p, so the recorded
# action stream drifts from true position -- exactly the InEKF premise -- while the
# observations, which reflect TRUE cells, carry the correction signal. The blind
# query phase then asks whether the map survived. Headroom is real: the best arm on
# clean Match-Query 128^2 is 0.870 against a 0.0625 chance rate.
#
# GATED BEFORE ANY GPU (validate_match_query.py, same gate code as the clean task,
# with the noise knob threaded through rather than a reimplementation -- rule 7):
#
#   p       chance   marginal  ngram1  ngram3  never_moved  drift (cells)
#   0.00    0.0625   0.0757    0.0437  0.0801  0.0892       0.00
#   0.05    0.0625   0.0735    0.0448  0.0535  0.0963       8.53
#   0.10    0.0625   0.0854    0.0594  0.0849  0.1204      13.05
#   0.25    0.0625   0.0785    0.0683  0.0915  0.0923      22.32
#
# Every shortcut gate stays at its clean-task level, and DRIFT rises 0 -> 22 cells
# on a 128 grid. p=0 drift is EXACTLY 0, which also confirms the env patch is inert
# by default -- separately verified byte-identical to the pre-patch code.
#
# QUERY STAYS CLEAN, deliberately. Scoring is keyed on the agent's TRUE cell; noisy
# query transitions would make that cell unknowable from the recorded stream, so the
# ceiling would fall to chance for every architecture. That is unanswerable, not
# harder. Explore-only is also the honest setting: odometry is unreliable while
# mapping, and the map is then queried.
#
# PRE-REGISTERED. Primary is Level15 - Vanilla at p=0.10, the contrast whose premise
# this task was built to supply, against the SAME contrast at p=0 in the same batch:
#   effect grows with p and clears its MDE -> the filter pays where its premise
#       holds, and every prior null was a premise failure, not a mechanism failure.
#   effect flat in p -> the InEKF does not buy drift correction even when drift is
#       the thing being measured, and the "stabilisation not inference" reading
#       survives its sharpest test.
# Secondary is the interaction (L15Loop - Loop) - (L15 - Vanilla) at p=0.10: last
# night it was unmeasured on a task with no premise, so this re-asks it with one.
#
# The recipe comes from step 1 (RECIPE_CHOICE.json). LR transfers; the epoch count
# does not transfer across tasks, so only the step-1 SCALE FACTOR is carried and
# convergence is RE-MEASURED here rather than assumed.
#
# 4 arms x 2 noise x 8 seeds = 64 runs, ~15 min each at 300 ep.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mq_noise"; mkdir -p "$R"
LOG="$REPO/mq_noise.log"; echo "mq-noise queued $(date)" > "$LOG"
A="train_var""iant"; B="train_match_""query"; MAXPG=5
SIZE=128; NOBS=16; TE=512; TQ=256; NB=48; BS=16; DM=128; NH=2

echo "$(date +%H:%M) waiting for the recipe batch" >> "$LOG"
until [ -f "$REPO/.recipe_power_done" ]; do sleep 120; done
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done

EP=300; LR=3e-4
if [ -f "$REPO/RECIPE_CHOICE.json" ]; then
  LR=$(python3 -c "import json;print(json.load(open('$REPO/RECIPE_CHOICE.json'))['lr'])")
  RE=$(python3 -c "import json;print(json.load(open('$REPO/RECIPE_CHOICE.json'))['epochs'])")
  [ "$RE" = "600" ] && EP=600
  echo "$(date +%H:%M) recipe from step 1: lr=$LR epochs=$EP" >> "$LOG"
else
  echo "$(date +%H:%M) NO RECIPE_CHOICE.json -- falling back to lr 3e-4, 300 ep" >> "$LOG"
fi

on_gpu(){ pgrep -u "$USER" -af "$B" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"; P="$3"; TAG="$4"
  OUT="$R/$TAG/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}_matchquery.pt" ] && { echo "skip $TAG $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $TAG $V s$SEED (p=$P) -> cuda:$GPU ($N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
    --epochs "$EP" --lr "$LR" --n-batches $NB --batch-size $BS --size $SIZE \
    --n-obs $NOBS --T-explore $TE --T-query $TQ --n-layers 1 --d-model $DM \
    --n-heads $NH --p-transition-noise "$P" --schedule cosine --fast-attn \
    --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${TAG}_${V}_s${SEED}.log" 2>&1 &
  sleep 8
}
for SEED in 0 1 2 3 4 5 6 7; do
  for V in Vanilla Level15 Looped Level15Looped; do
    launch "$V" "$SEED" 0.0  p0
    launch "$V" "$SEED" 0.10 p010
  done
done
wait
N=$(find "$R" -name '*_matchquery.pt' | wc -l)
echo "$(date +%H:%M) $N/64 checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_mq_noise --repo "$REPO" --runs-dir "$R" \
  --out "$REPO/MQ_NOISE_2X2.md" >> "$LOG" 2>&1
touch "$REPO/.mq_noise_done"; echo "$(date) DONE" >> "$LOG"
