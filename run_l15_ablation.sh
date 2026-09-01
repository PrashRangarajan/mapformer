#!/usr/bin/env bash
# WHAT IS LEVEL 1.5 ACTUALLY MADE OF? -- the ablation set at n=5.
#
# THE CLAIM UNDER TEST. The project's standing account is that the InEKF's win is
# "stabilisation and token-type gating, NOT Bayesian inference". Three separate
# lines now support the 'not inference' half: R_t learns to be high on aliased
# observations (the filter gates its own measurements off); a capacity control ties
# Level15 on lm200; and the depth-axis refine-theta variant, tested in the regime
# built for it, gives -0.001/-0.011/+0.005 at T=128 and +0.006/+0.003/-0.005 at
# T=512 with no slope in noise.
#
# What is NOT established is the positive half -- which structural piece does the
# work. The whole decomposition rests on ONE SEED per arm (RESULTS_PAPER.md):
#
#     Level15    1.000 / 0.993      wrap + measurement + per-token R
#     L15_DARE   1.000 / 0.992      same, but Pi fixed by DARE not learned
#     L15_NoMeas 0.904 / 0.831      z == 0: the WRAP ALONE, a pure bounded clamp
#     L15_NoCorr 0.940 / 0.833      correction zeroed == vanilla MapFormer
#     L15_ConstR 0.795 / 0.672      wrap + measurement, NO per-token gate
#
# Two claims hang off that table and both deserve more than n=1:
#   1. "Level15 does NOT reduce to clamping theta" -- rests on NoMeas (0.831)
#      falling well short of Level15 (0.993).
#   2. "The token-type gate is load-bearing, and removing it is WORSE THAN DOING
#      NOTHING" -- rests on ConstR (0.672) < NoCorr (0.833). That is a strong,
#      surprising claim from a single draw.
#   3. L15_DARE == Level15 would say the principled Kalman gain is irrelevant --
#      the learned scalar does just as well.
#
# PRE-REGISTERED:
#   A. NoMeas << Level15 and ConstR < NoCorr -> the published decomposition holds;
#      Level15 = bounded state + token-type gate, and neither is inference.
#   B. NoMeas ~ Level15 -> it DOES reduce to clamping theta, and the measurement
#      head and R_t head are dead weight.
#   C. ConstR ~ NoCorr -> the 'worse than nothing' result was a single-seed
#      artifact; the gate is merely inert, not harmful.
#   D. NoCorr ~ Level15 -> the whole correction is inert on this task and the
#      original Level15 advantage was something else (capacity, RNG, schedule).
#
# Clean config only: the lm200 column of that table is under the 2026-07-16
# convergence retraction and must not be used.
#
# 6 arms x 5 seeds = 30 runs, ~42 min each at 6-way concurrency -> ~3.5 h.
# 300 epochs with warmup+cosine (rule 10 -- the published numbers used
# LinearLR-from-step-one, so this is a retrain, not a comparison to them).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/l15_ablation"; mkdir -p "$R/p0"
LOG="$REPO/l15_ablation.log"; echo "l15-ablation start $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"; MAXPG=3
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
for SEED in 0 1 2 3 4; do
  for V in Vanilla Level15 L15_NoMeas L15_NoCorr L15_ConstR L15_DARE; do launch "$V" "$SEED"; done
done
wait
N=$(find "$R" -name '*.pt' | wc -l); echo "$(date +%H:%M) $N/30 checkpoints" >> "$LOG"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Level15 L15_NoMeas L15_NoCorr L15_ConstR L15_DARE \
  --noises 0.0 --seeds 0 1 2 3 4 --lengths 128 512 1024 --n-trials 100 \
  --device cuda:0 --out "$REPO/L15_ABLATION.md" >> "$LOG" 2>&1
[ -f "$REPO/L15_ABLATION.md" ] && echo "$(date +%H:%M) eval OK" >> "$LOG" \
  || echo "$(date +%H:%M) EVAL FAILED" >> "$LOG"
touch "$REPO/.l15_ablation_done"; echo "$(date) DONE" >> "$LOG"
