#!/usr/bin/env bash
# DOES REFINING THETA HELP WHEN THE ACTIONS ARE NOISY?
#
# THE DESIGN ERROR THIS CORRECTS. The refine-theta loop was first tested on
# Match-Query. Actions there are CLEAN and the query phase is BLIND, so neither
# half of the InEKF premise holds: with clean actions cumsum(delta) has no drift
# to correct, and with observations withheld there is nothing to correct WITH.
# This repo had already measured that for the sequence-axis correction --
# "Match-Query (blind) 0.876 vs 0.888, no advantage, nothing to correct with" --
# so the depth-axis null replicated a known negative rather than discovering one.
#
# THE RIGHT REGIME. p_action_noise corrupts the action RECORD while the agent
# still moves per the true action, so the model's path integral drifts away from
# true position while the OBSERVATIONS still reflect true position. That is
# exactly the setup a Kalman-style correction is for. (Equivalent to a
# stochastic-transition MDP for uniform policies; use that framing in writeups.)
#
# ARMS -- all path-integrated, torus paper task, 1 layer, d=128:
#   Vanilla         no loop, no correction                       (baseline)
#   Looped x4       loop, theta computed ONCE                    (control)
#   LoopedRefine x4 loop, theta corrected each pass              (treatment)
#   Level15         the SEQUENCE-axis InEKF correction           (incumbent)
# Level15 is included because it is the mechanism that demonstrably works under
# noise here (+11pp at T=512 in the project log). A depth-axis correction should
# be measured against it, not against nothing.
#
# NOISE LEVELS 0.0 / 0.10 / 0.25 -- three points, because two make a line and a
# line is not a trend (rule 5 corollary). 0.10 matches the prior Level15 result.
#
# PRE-REGISTERED. The claim is not "refine beats fixed"; it is that the gain
# GROWS WITH NOISE:
#   positive slope in (refine - fixed) vs noise -> the correction mechanism is
#       real, and the Match-Query null was a wrong-regime artifact.
#   flat slope, gain > 0 at all levels -> the loop's benefit is ITERATION on any
#       input; refinement is not doing correction work.
#   flat slope, gain ~ 0 -> refinement adds nothing anywhere, and the depth-axis
#       version of the Kalman idea is dead. Combined with the sequence-axis
#       finding (stabilisation, not inference) that is a coherent negative.
#   Level15 > LoopedRefine at high noise -> correction belongs on the sequence
#       axis, where the state actually evolves, not on the depth axis.
#
# 4 arms x 3 noise x 3 seeds = 36 runs. Trained AND evaluated under the same
# noise, on a held-out map, at train length (128) and OOD length (512) -- drift
# compounds with length, so OOD is where correction should show if anywhere.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/noise_refine"; mkdir -p "$R"
LOG="$REPO/noise_refine.log"; echo "noise-refine start $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"
MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"; P="$3"; TAG="$4"
  OUT="$R/$TAG/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V $TAG s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) $V $TAG s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --n-steps $T --n-layers 1 \
    --n-heads $NH --d-model $DM --n-landmarks 0 --p-action-noise "$P" \
    --schedule cosine --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${TAG}_${V}_s${SEED}.log" 2>&1 &
  sleep 8
}
# noise level varies fastest so a partial run still spans the axis under test
for SEED in 0 1 2; do
  for NP in "0.0 p0" "0.10 p01" "0.25 p025"; do
    set -- $NP; P=$1; TAG=$2
    for V in Vanilla Looped LoopedRefine Level15; do launch "$V" "$SEED" "$P" "$TAG"; done
  done
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/36 checkpoints" >> "$LOG"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Looped LoopedRefine Level15 --noises 0.0 0.10 0.25 \
  --seeds 0 1 2 --lengths 128 512 --device cuda:0 \
  --out "$REPO/NOISE_REFINE.md" >> "$LOG" 2>&1
touch "$REPO/.noise_refine_done"; echo "$(date) DONE" >> "$LOG"
