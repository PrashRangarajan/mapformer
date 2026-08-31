#!/usr/bin/env bash
# DOES REFINING THE POSITION ESTIMATE EACH PASS BEAT RE-READING A FIXED ONE?
#
# Match-Query at n=8 showed loop x path-integration composes super-additively
# (+0.414 on the path-int arm vs +0.099 on index, interaction +0.315). But that
# loop computes theta ONCE and re-reads it every pass. The natural follow-on is a
# loop that CARRIES and CORRECTS a position estimate -- structurally this repo's
# InEKF work moved from the sequence axis to the depth axis.
#
#     theta_0 = omega * cumsum(action_to_lie(emb))
#     x       = block(x, cos theta, sin theta)
#     theta   = theta_0 + gate * tanh(refine(x))
#
# Verified before spending anything: at gate=0 the model is BIT-IDENTICAL to
# Looped (max diff 0.00e+00); at gate=0.5 the output moves (0.035); and the gate
# gradient AT gate=0 is 1.9e-03, so the no-op init is escapable rather than a
# permanent trap. +385 params on 204,630 (0.19%).
#
# PRE-REGISTERED:
#   refine > Looped beyond MDE -> iterative position refinement adds something a
#       fixed theta cannot. The InEKF idea works on the depth axis, where it did
#       not on the sequence axis.
#   refine ~ Looped -> the loop's benefit is ITERATION alone. Consistent with this
#       project's standing finding that the Kalman win was stabilisation and
#       token-type gating, NOT inference -- now shown to hold on a second axis.
#   refine < Looped -> refinement actively hurts, the failure mode Level15EM hit.
#
# ALSO DIAGNOSTIC, regardless of outcome: the LEARNED GATE. If it stays near zero
# the model declined to refine, which is a mechanism answer rather than a null --
# the same tell that exposed the learnable-beta result as a red herring (learned
# betas barely moved from init, so a 1.2x sharpening could not explain +12pp).
#
# Both arms retrained in ONE batch (rule 3), n=8, identical settings.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/refine"; mkdir -p "$R"
LOG="$REPO/refine.log"; echo "refine start $(date)" > "$LOG"
EP=300; NB=48; BS=16; SZ=128; NOBS=16; TE=512; TQ=256; DM=128; NH=2
A="train_match_""query"
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done
MAXPG=3
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; SEED="$2"
  OUT="$R/$V/s$SEED"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 45
  done
  echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --size $SZ --n-obs $NOBS \
    --T-explore $TE --T-query $TQ --eval-query 256 512 --n-layers 1 \
    --d-model $DM --n-heads $NH --schedule cosine --fast-attn \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
  sleep 15
}
for SEED in 0 1 2 3 4 5 6 7; do
  launch Looped "$SEED"; launch LoopedRefine "$SEED"
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/16 checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_refine --runs-dir "$R" --out "$REPO/REFINE_RESULTS.md" >> "$LOG" 2>&1
touch "$REPO/.refine_done"; echo "$(date) DONE" >> "$LOG"
