#!/usr/bin/env bash
# TEST OF THE VISITS-PER-CELL HYPOTHESIS.
#
# THE CLAIM. The position effect switches on when a scored revisit has ~one prior
# exemplar in context: with many prior visits attention has plenty to match
# against, with one it must localise precisely.
#
# WHY THE EXISTING DATA CANNOT TEST IT. probe_visits_per_cell.py measured the
# realised distribution rather than assuming T/n_occupied (my arithmetic said
# 16/4/1; the truth is 8.64/4.61/3.05, because the walk is directed and revisits
# far more than uniform coverage implies). Across grid 8/16/32 at T=512 the three
# candidate predictors move together:
#     grid 8   46 distinct,  8.64 prior,  32 occupied  -> -0.010
#     grid 16  96 distinct,  4.61 prior, 128 occupied  -> +0.015
#     grid 32 158 distinct,  3.05 prior, 512 occupied  -> +0.374
# so no pair separates them. Note also the shape: a 1.8x drop in prior visits
# (8.64->4.61) produced NOTHING, then a 1.6x drop (4.61->3.05) produced
# everything. Threshold, not dose-response -- for every predictor equally.
#
# THE INSTRUMENT. Vary T at fixed grid to decouple them. Two new conditions, each
# MATCHED ON DISTINCT CELLS VISITED with a condition already measured:
#
#   A  grid 32, T=128    48 distinct, 1.95 prior, 512 occupied
#      vs grid 8, T=512  46 distinct, 8.64 prior,  32 occupied  (= -0.010)
#      Distinct matched; prior visits differ 4.4x; map extent differs 16x.
#        large effect -> NOT distinct-cells-visited. Either prior visits or map
#                        extent, which this pair cannot separate from each other.
#        ~zero        -> distinct cells visited is what matters, and the
#                        visits-per-cell hypothesis is DEAD (its prior count is
#                        the lowest anywhere in the study, 1.95).
#
#   B  grid 16, T=1024  153 distinct, 6.20 prior, 128 occupied
#      vs grid 32, T=512 158 distinct, 3.05 prior, 512 occupied (= +0.374)
#      Distinct matched; prior visits differ 2.0x; map extent differs 4x.
#        ~+0.374 -> distinct cells visited drives it; prior visits and map extent
#                   are both ruled out.
#        ~zero   -> distinct is ruled out; prior visits or map extent survive.
#
# Together A and B triangulate: A alone cannot separate prior-visits from map
# extent, B alone cannot either, but the PAIR is inconsistent with any single one
# of the three surviving on its own unless the results line up one specific way.
# All four outcomes are enumerated in agg_visits.py before the runs finish.
#
# Trained AND evaluated at the training length, so no arm is scored out of
# distribution. fast-attn throughout (licensed: +0.392 vs +0.374 reference).
# A is cheap (seq 256); B is seq 2048 and is the expensive half -- A runs first so
# a surprise shows up in an hour rather than ten.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/visits_test"; mkdir -p "$R"
LOG="$REPO/visits_test.log"; echo "visits test queued $(date)" > "$LOG"
NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
A="train_mini""world"

echo "$(date +%H:%M) waiting for the looped pilot to finish" >> "$LOG"
until [ -f "$REPO/.looped_pilot_done" ]; do sleep 180; done
B2="train_var""iant"
while [ "$(pgrep -u "$USER" -f "$A" | wc -l)" -gt 0 ] || \
      [ "$(pgrep -u "$USER" -f "$B2" | wc -l)" -gt 0 ]; do sleep 120; done
echo "$(date +%H:%M) GPUs clear" >> "$LOG"

FREE_MB=$(df -Pm "$REPO" | awk 'NR==2{print $4}')
[ "$FREE_MB" -lt 3000 ] && { echo "REFUSING: only ${FREE_MB} MB free" >> "$LOG"; \
  touch "$REPO/.visits_test_done"; exit 1; }

# serial buffer prebuild (EGL contexts saturate a GPU otherwise)
python3 -u -m mapformer.prebuild_buffers --grid-size 32 --n-obs 256 --seeds 0 1 2 \
  --n-steps 128 --buffer-size $NBUF --eval-trials $ETRIALS --eval-lengths 128 \
  --n-workers $NW --oracle >> "$LOG" 2>&1
python3 -u -m mapformer.prebuild_buffers --grid-size 16 --n-obs 64 --seeds 0 1 2 \
  --n-steps 1024 --buffer-size $NBUF --eval-trials $ETRIALS --eval-lengths 1024 \
  --n-workers $NW --oracle >> "$LOG" 2>&1

MAXPG=4
on_gpu(){ pgrep -u "$USER" -af "$A" 2>/dev/null | grep -c -- "--device cuda:$1" || true; }
launch(){   # variant seed grid n_obs T outdir
  V="$1"; SEED="$2"; G="$3"; NOBS="$4"; TT="$5"; OUT="$6"
  mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip $V g$G T$TT s$SEED" >> "$LOG"; return; }
  GPU=""
  while :; do
    N0=$(on_gpu 0); N1=$(on_gpu 1)
    # balance, NOT fill-first: a fill-first picker silently becomes a single-GPU
    # scheduler whenever the job count is <= MAXPG (it idled a 4090 for 3 h).
    if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$N0" ] && [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    if [ "$N0" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "$N1" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 60
  done
  echo "$(date +%H:%M) $V g$G n_obs=$NOBS T=$TT s$SEED -> cuda:$GPU (load $N0/$N1)" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --fast-attn --grid-size "$G" --n-obs "$NOBS" --n-steps "$TT" --buffer-size $NBUF \
    --epochs $EP --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL \
    --n-heads $NH --n-workers $NW --schedule cosine --eval-trials $ETRIALS \
    --eval-lengths "$TT" --device "cuda:$GPU" --output-dir "$OUT" \
    > "$R/${V}_g${G}_T${TT}_s${SEED}.log" 2>&1 &
  sleep 30
}

# A first (cheap, seq 256): a surprise shows up in an hour, not ten
echo "$(date +%H:%M) === A: grid 32, T=128 (48 distinct, 1.95 prior) ===" >> "$LOG"
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 32 256 128 "$R/A_g32_T128/s$SEED"; done
done
wait
echo "$(date +%H:%M) === B: grid 16, T=1024 (153 distinct, 6.20 prior) ===" >> "$LOG"
for SEED in 0 1 2; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 16 64 1024 "$R/B_g16_T1024/s$SEED"; done
done
wait

echo "$(date +%H:%M) $(find "$R" -name '*_oracle.pt' | wc -l)/12 checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_visits --runs-dir "$R" --out "$REPO/VISITS_TEST.md" >> "$LOG" 2>&1
touch "$REPO/.visits_test_done"; echo "$(date) DONE" >> "$LOG"
