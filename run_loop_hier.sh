#!/usr/bin/env bash
# DOES THE LOOP x HIERARCHY PAIRING KEEP ACCURACY? The compute half is already
# measured; this is the half that decides whether the saving is free.
#
# THE PAIRING. The two mechanisms save different resources and neither saves the
# other's: the loop saves PARAMETERS and costs COMPUTE (passes cost passes),
# hierarchy saves COMPUTE and costs PARAMETERS (a 3-block scaffold). Measured at
# L=2048, batch 64, each arm alone on an idle device (LOOP_HIER_COMPUTE.md):
#
#   arm                   params            ms/step            peak MiB
#   HourglassFlat3        596,034           280.97             20,552
#   Hourglass_k2          596,034           216.62  (-22.9%)   16,468  (-19.9%)
#   LoopedHourglassFlat   199,490 (-66.5%)  281.20  ( +0.1%)   20,547
#   LoopedHourglass       199,490 (-66.5%)  216.77  (-22.8%)   16,463  (-19.9%)
#
# The savings COMPOSE EXACTLY -- sharing costs nothing in time, hierarchy costs
# nothing in parameters. So the only question left is accuracy.
#
# TRAINED AT L=512, NOT L=16. The compute saving is length-dependent and does not
# exist at short lengths: at L=16 the hierarchical arm is 12% SLOWER and at L=128
# 2% slower, both overhead-dominated, and the saving only appears from about L=512.
# Testing accuracy at a length where the compute claim is false would answer a
# question nobody asked. Eval extends to L=2048 where the saving is largest.
#
# WHAT WE ALREADY KNOW GOING IN. Hierarchy buys NOTHING in accuracy on parity: at
# L=128 trained-at-16 the index row was +0.001 with an MDE of 0.006, a tight null
# (HIER_PARITY.md). So the honest hypothesis here is NOT that the combination is
# better -- it is that the combination is CHEAPER AT THE SAME ACCURACY. A null on
# accuracy is the SUCCESS case for this run, which is why the compute numbers were
# measured first: without them a null here would be uninterpretable.
#
# PRE-REGISTERED:
#   all four arms within noise -> the pairing is FREE: 66.5% fewer parameters and
#       22.8% less time and memory at equal accuracy. Both savings, no cost.
#   LoopedHourglass below the unshared arms -> sharing costs accuracy at this
#       length, and the parameter saving is not free after all.
#   LoopedHourglass below LoopedHourglassFlat specifically -> the POOLING costs
#       accuracy once weights are shared, i.e. the two savings interfere even
#       though their resources do not.
#
# 4 arms x 12 seeds = 48 runs at L=512.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/loop_hier"; mkdir -p "$R"
LOG="$REPO/loop_hier.log"; echo "loop-hier start $(date)" > "$LOG"
A="train_algo""rithmic"; MAXPG=5
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  for SPEC in "HourglassFlat3 1" "Hourglass_k2 1" "LoopedHourglassFlat 1" "LoopedHourglass 1"; do
    set -- $SPEC; V="$1"; NL="$2"
    OUT="$R/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}_parity.json" ] && continue
    GPU=""
    while :; do
      N0=$(on_gpu 0); N1=$(on_gpu 1)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
      sleep 10
    done
    echo "$(date +%H:%M:%S) $V L$NL s$SEED -> cuda:$GPU" >> "$LOG"
    python3 -u -m mapformer.train_algorithmic --variant "$V" --task parity \
      --seed "$SEED" --epochs 150 --n-batches 30 --batch-size 32 \
      --train-length 512 --eval-lengths 512 1024 2048 --lr 1e-3 \
      --d-model 128 --n-heads 2 --n-layers "$NL" --schedule cosine \
      --device "cuda:$GPU" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 3
  done
done
wait
N=$(find "$R" -name '*.json' | wc -l); echo "$(date +%H:%M) $N/48 results" >> "$LOG"
python3 -u -m mapformer.agg_loop_hier --runs-dir "$R" \
  --out "$REPO/LOOP_HIER_PARITY.md" >> "$LOG" 2>&1
touch "$REPO/.loop_hier_done"; echo "$(date) DONE" >> "$LOG"
