#!/usr/bin/env bash
# THE LOOPED LITERATURE'S OWN TASKS: does path integration help there?
#
# WHY. Everything this project knows about looping is from navigation. Match-Query
# has no known iterative solution -- answering it is compute-position, match, read,
# a few COMPOSED operations -- so the loop's +0.414 there is depth substitution,
# i.e. the ALBERT/MoR parameter-efficiency claim. The looped literature's own claim
# is different: looping supplies a variable number of ALGORITHMIC STEPS, and it is
# benchmarked on tasks with a known iterative solution under LENGTH GENERALIZATION
# (arXiv 2409.15647: parity, binary addition, copying).
#
# TWO TASKS, pointing opposite ways, so "path integration helps" cannot be a claim
# about our pipeline:
#   PARITY  the canonical iterative task, and where path integration has a SHARP
#           mechanistic prediction. MapFormer's angle is theta = omega *
#           cumsum(Delta(x_t)) wrapped mod 2*pi; parity is a running sum mod 2.
#           Delta = pi for '1' and 0 for '0' puts the answer in the rotation.
#   COPY    positional retrieval, no iterative structure, nothing to accumulate.
#           Index position should suffice and path integration should buy little.
#
# FIVE ARMS: the 2x2 of {index RoPE, path integration} x {flat, loop x4}, plus
# LoopedSampled -- the loop count drawn per training batch. That arm is here
# because the literature's length-generalization claim is specifically about
# looping with an ADAPTIVE number of steps, and our own torus data says a FIXED
# count fails out of distribution (T=1024: Looped 0.730 vs Vanilla 0.749) while
# sampling recovers it (0.816 -> 0.915 at T=512).
#
# GATED BEFORE TRAINING (ALGORITHMIC_GATES.md): marginal, n-gram orders 1/2/3/5,
# echo-input and repeat-prev all sit at the measured chance rate at every length,
# worst excess +0.0122. Standing rule 1 -- four planner tasks in this repo were
# voided by exactly this check AFTER the GPU had been spent.
#
# PRE-REGISTERED:
#   H1 path integration helps PARITY much more than COPY -> the effect is about the
#      task's structure, not about our pipeline. If it helps both equally, the
#      mechanism story is wrong and something generic is going on.
#   H2 the loop improves LENGTH GENERALIZATION (the ratio of L=128 to L=16
#      accuracy), which is the literature's actual claim and one this project has
#      never tested on its own terms.
#   H3 LoopedSampled beats fixed Looped at long L. This is the sharpest test of the
#      adaptivity claim, and it is the one place our navigation data already agrees
#      with the field.
#
# Train at L=16 and evaluate to L=256, 8 seeds. 5 arms x 2 tasks x 8 seeds = 80
# runs at ~75 s each -- about 15 minutes at 10-way concurrency.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/algorithmic"; mkdir -p "$R"
LOG="$REPO/algorithmic.log"; echo "algorithmic start $(date)" > "$LOG"
EP=300; NB=50; BS=128; LTR=16; DM=128; NH=2; LR=1e-3
A="train_algo""rithmic"; MAXPG=5
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){ V="$1"; TASK="$2"; SEED="$3"
  OUT="$R/$TASK/${V}_s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}_${TASK}.json" ] && { echo "skip $TASK $V s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 10
  done
  echo "$(date +%H:%M:%S) $TASK $V s$SEED -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_algorithmic --variant "$V" --task "$TASK" \
    --seed "$SEED" --epochs $EP --n-batches $NB --batch-size $BS \
    --train-length $LTR --eval-lengths 16 32 64 128 256 --lr $LR \
    --d-model $DM --n-heads $NH --n-layers 1 --schedule cosine \
    --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${TASK}_${V}_s${SEED}.log" 2>&1 &
  sleep 3
}
for SEED in 0 1 2 3 4 5 6 7; do
  for TASK in parity copy; do
    for V in RoPE Vanilla RoPELooped Looped LoopedSampled; do
      launch "$V" "$TASK" "$SEED"
    done
  done
done
wait
N=$(find "$R" -name '*.json' | wc -l); echo "$(date +%H:%M) $N/80 results" >> "$LOG"
python3 -u -m mapformer.agg_algorithmic --runs-dir "$R" \
  --out "$REPO/ALGORITHMIC_RESULTS.md" >> "$LOG" 2>&1
touch "$REPO/.algorithmic_done"; echo "$(date) DONE" >> "$LOG"
