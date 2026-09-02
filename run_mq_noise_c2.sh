#!/usr/bin/env bash
# STEP 2, TAKE 2: the same 2x2 with a recipe that lets the arms CONVERGE.
#
# WHY A SECOND RUN. Take 1 answered the primary question -- Level15 - Vanilla is
# +0.035 at p=0 and +0.038 at p=0.10, i.e. FLAT in drift, which is the signature of
# a mechanism that is not doing correction. But it answered it between two arms
# that were both far from converged. Final MATCH loss at p=0.10 (chance is 2.77):
#
#   arm              p=0.0                p=0.10
#   Vanilla          1.595 (0.43-2.75)    2.526 (2.39-2.60)
#   Level15          1.432 (0.44-2.07)    2.367 (2.29-2.50)
#   Looped           0.457 (0.01-0.86)    2.067 (1.86-2.21)
#   Level15Looped    0.892 (0.20-1.39)    2.198 (2.10-2.44)
#
# Everything is below chance, so the models learn -- but only just at p=0.10,
# against 0.46 for the best arm at p=0. Rule 10 forbids reading a comparison
# between unconverged arms: it measures which arm escapes the plateau fastest, not
# which can solve the task. The primary contrast is exactly such a comparison.
#
# THE FIX comes from step 1, which take 1 could not use because a broken Pareto
# rule handed it C0. lr 1e-3 cut the failing arm's seed sd by 3.5x on the torus
# (T=512: 0.096 -> 0.028) with a HIGHER mean at every length. Here it is paired
# with 600 epochs because the evidence in THIS task says the binding problem is
# under-training, not basin selection.
#
# That combination is step 1's C2, and step 1 found C2 had the WORST torus OOD sd
# (0.158 at T=1024 against C0's 0.110). Taking it anyway is a deliberate call: the
# torus objection is about long-OOD variance in a regime that converges, while the
# problem here is arms that do not converge at all. If C2 reproduces its torus
# variance problem here, the convergence table will show it and the run says so.
#
# WHAT IS NOT CHANGED. Two noise levels, not three. A third dose point was the
# other obvious upgrade, and it is dropped on purpose: the effect is currently
# FLAT, and a third point on a flat line buys much less than making the two
# existing points legitimate. If take 2 shows a non-flat effect, the dose-response
# run becomes worth its 5 hours; until then it is not.
#
# PRE-REGISTERED, unchanged from take 1: Level15 - Vanilla at p=0.10 against the
# same contrast at p=0, in one batch. The effect must GROW with drift. Take 1's
# change was +0.003 across 0 -> 13 cells of drift; if that survives on converged
# arms, "stabilisation, not inference" has passed the sharpest test available.
#
# 4 arms x 2 noise x 8 seeds = 64 runs at 600 ep.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/mq_noise_c2"; mkdir -p "$R"
LOG="$REPO/mq_noise_c2.log"; echo "mq-noise queued $(date)" > "$LOG"
A="train_var""iant"; B="train_match_""query"; MAXPG=5
SIZE=128; NOBS=16; TE=512; TQ=256; NB=48; BS=16; DM=128; NH=2

echo "$(date +%H:%M) waiting for the first mq_noise batch to clear" >> "$LOG"
until [ -f "$REPO/.mq_noise_done" ]; do sleep 120; done
while [ "$(pgrep -u "$USER" -f "$B" | wc -l)" -gt 0 ]; do sleep 60; done
EP=600; LR=1e-3
echo "$(date +%H:%M) recipe: lr=$LR epochs=$EP (see header for why)" >> "$LOG"

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
  --out "$REPO/MQ_NOISE_2X2_C2.md" >> "$LOG" 2>&1
touch "$REPO/.mq_noise_c2_done"; echo "$(date) DONE" >> "$LOG"
