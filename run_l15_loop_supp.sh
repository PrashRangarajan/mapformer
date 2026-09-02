#!/usr/bin/env bash
# SUPPLEMENT to run_l15_loop_2x2.sh -- adds concurrency without disturbing it.
#
# WHY. Measured split of one epoch: trajectory generation 9.1 s, model fwd+bwd
# 0.5 s (Vanilla) / 2.5 s (Level15Looped). Data generation is 79-95% of the time,
# it is single-threaded Python, and each job therefore pegs exactly ONE core.
# With 8 jobs running, 24 of 32 cores are idle and the GPUs sit near 70% util on
# what is really a trivial compute load. --fast-attn is NOT the answer at this
# size: it gives 1.28x on the compute, i.e. ~1% of the epoch, and it would also
# be a mid-batch code-path change against the 10 runs already finished.
#
# DESIGN. This does NOT touch the running script (rule 16: bash reads by byte
# offset, editing a live script can resume it mid-token). It is a second scheduler
# that works the seed list from the BACK while the main one works from the front,
# so they consume the same queue from opposite ends and meet in the middle.
#
# TWO GUARDS, because the main script's own guard only covers FINISHED runs:
#   1. skip if the checkpoint already exists (same as the main script)
#   2. skip if a process is already training that exact --variant + --seed
# Together these make a collision impossible rather than merely unlikely.
#
# CONCURRENCY CAP is on TOTAL jobs per GPU, counting the main script's, at 7.
# Per-job GPU memory measured at 1.1-3.4 GB (looped arms are the heavy ones), so
# 7 x ~2.9 GB ~= 20 GB of 24.5 GB. Eight would be ~23 GB and risks an OOM that
# would silently kill an arm.
#
# EVAL. The main script runs the eval when ITS queue empties, which will be before
# this one finishes, so that eval will be short some seeds. This script therefore
# re-runs the eval and the aggregation itself, gated on all 60 checkpoints being
# present. Last writer wins and it is the complete one.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/l15_loop_2x2"
LOG="$REPO/l15_loop_supp.log"; echo "supplement start $(date)" > "$LOG"
EP=300; NB=98; BS=128; T=128; DM=128; NH=2
A="train_var""iant"; CAP=7
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
in_flight(){ pgrep -u "$USER" -af "$A" 2>/dev/null \
             | grep -q -- "--variant $1 --seed $2 "; }

launch(){ V="$1"; SEED="$2"
  OUT="$R/p0/${V}_s${SEED}"
  [ -f "$OUT/${V}.pt" ] && { echo "skip(done) $V s$SEED" >> "$LOG"; return; }
  in_flight "$V" "$SEED" && { echo "skip(in-flight) $V s$SEED" >> "$LOG"; return; }
  mkdir -p "$OUT"
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$CAP" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$CAP" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$CAP" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$CAP" ]; then GPU=1; break; fi
    sleep 30
  done
  # re-check both guards after the wait: the main script may have claimed it
  [ -f "$OUT/${V}.pt" ] && { echo "skip(done,late) $V s$SEED" >> "$LOG"; return; }
  in_flight "$V" "$SEED" && { echo "skip(in-flight,late) $V s$SEED" >> "$LOG"; return; }
  echo "$(date +%H:%M) $V s$SEED -> cuda:$GPU (total load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
    --epochs $EP --n-batches $NB --batch-size $BS --n-steps $T --n-layers 1 \
    --n-heads $NH --d-model $DM --n-landmarks 0 --schedule cosine \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
  sleep 8
}

# back of the queue, descending -- the main script runs 0..11 ascending
for SEED in 11 10 9 8 7 6 5 4; do
  for V in Vanilla Level15 Looped Level15Looped LoopedSampled; do launch "$V" "$SEED"; done
done
wait

# wait for the main script's own children too, so the count below is final
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ]; do sleep 60; done

N=$(find "$R" -name '*.pt' | wc -l)
echo "$(date +%H:%M) all training stopped; $N/60 checkpoints" >> "$LOG"
if [ "$N" -ne 60 ]; then
  echo "INCOMPLETE -- not overwriting the main script's eval. Missing arms:" >> "$LOG"
  for S in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for V in Vanilla Level15 Looped Level15Looped LoopedSampled; do
      [ -f "$R/p0/${V}_s${S}/${V}.pt" ] || echo "  MISSING $V s$S" >> "$LOG"
    done
  done
  touch "$REPO/.l15_loop_supp_done"; exit 1
fi

echo "$(date +%H:%M) re-running eval on the complete set" >> "$LOG"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" \
  --variants Vanilla Level15 Looped Level15Looped LoopedSampled \
  --noises 0.0 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --lengths 128 512 1024 \
  --n-trials 100 --device cuda:0 --out "$REPO/_L15_LOOP_RAW.md" >> "$LOG" 2>&1
if [ -f "$REPO/_L15_LOOP_RAW.json" ]; then
  python3 -u -m mapformer.agg_l15_loop --repo "$REPO" \
    --json "$REPO/_L15_LOOP_RAW.json" --runs-dir "$R" \
    --out "$REPO/L15_LOOP_2X2.md" >> "$LOG" 2>&1
  echo "$(date +%H:%M) aggregation written" >> "$LOG"
else
  echo "$(date +%H:%M) EVAL FAILED -- no json" >> "$LOG"
fi
touch "$REPO/.l15_loop_supp_done"; echo "$(date) DONE" >> "$LOG"
