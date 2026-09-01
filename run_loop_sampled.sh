#!/usr/bin/env bash
# CAN THE LOOP'S LENGTH TRADE-OFF BE TRAINED AWAY?
#
# WHAT PROMPTED THIS. Under action noise the loop is a large win at TRAINING length
# (+0.138 at p=0.10, +0.205 at p=0.25, t>8) and a large LOSS out of distribution:
# the two looped arms lose 0.24 going 128 -> 512 while Vanilla and Level15 lose
# 0.06. But an eval-only sweep showed the damage is the ITERATION COUNT, not the
# architecture: with the SAME trained weights, T=512 accuracy peaks at 2 passes
# (0.794) and falls monotonically to 0.766 at 6, while T=128 rises monotonically to
# 1.000 at 4. The fixed count is a length-specific choice baked in at training time.
#
# So the trade-off may not be fundamental. `LoopedSampled` draws the count from
# {2..6} per training batch, making it a runtime knob the model tolerates across
# its range. Param-identical to Looped (204,373 both); verified to sample in
# training and be deterministic at eval.
#
# PRE-REGISTERED:
#   A. Sampled matches fixed-4 at T=128 AND beats it at T=512
#      -> the trade-off is a training artifact and is trainable away.
#   B. Sampled is WORSE at T=128 but better at T=512
#      -> real trade-off; sampling buys extrapolation with peak performance.
#   C. Sampled == fixed everywhere -> sampling does nothing; the count is not
#      what the model was over-fitting to.
#   D. Flatter curve, no better peak at any length -> robustness without gain,
#      which is still useful if the deployment length is unknown.
#
# The measurement is the SHAPE of accuracy vs (length x loops-at-eval), not one
# number -- the claim is about the curve flattening.
#
# 3 arms x 2 noise x 5 seeds = 30 runs, ~3 h. Vanilla is the no-loop reference.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/loop_sampled"; mkdir -p "$R"
LOG="$REPO/loop_sampled.log"; echo "loop-sampled start $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"; MAXPG=3
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
for SEED in 0 1 2 3 4; do
  for NP in "0.0 p0" "0.10 p01"; do
    set -- $NP; P=$1; TAG=$2
    for V in Vanilla Looped LoopedSampled; do launch "$V" "$SEED" "$P" "$TAG"; done
  done
done
wait
N=$(find "$R" -name '*.pt' | wc -l); echo "$(date +%H:%M) $N/30 checkpoints" >> "$LOG"
[ "$N" -lt 30 ] && echo "WARNING: incomplete" >> "$LOG"
python3 -u -m mapformer.eval_loop_sweep --runs-dir "$R" \
  --variants Vanilla Looped LoopedSampled --noises 0.0 0.10 --seeds 0 1 2 3 4 \
  --lengths 128 512 1024 --loops 1 2 3 4 6 --n-trials 80 --device cuda:0 \
  --out "$REPO/LOOP_SAMPLED.md" >> "$LOG" 2>&1
# verify the artifact exists rather than inferring success from no crash
[ -f "$REPO/LOOP_SAMPLED.md" ] && echo "$(date +%H:%M) eval OK" >> "$LOG" \
  || echo "$(date +%H:%M) EVAL FAILED -- see log" >> "$LOG"
touch "$REPO/.loop_sampled_done"; echo "$(date) DONE" >> "$LOG"
