#!/usr/bin/env bash
# Does r=4 remove the same failure mode the loop does, on Match-Query?
#
# WHY NOW. The D x r batch is separating rank's two halves, and the OPTIMISATION
# half is the one surviving (DXR_PRELIM.md: every D shows a rank effect, and
# r=D+2 beats r=D at every D, while the r=2 deficit does NOT grow with D). The
# optimisation account says r=2 learns a SKEWED basis -- opposition 0.495,
# |cos(N,E)| 0.783 -- and that the extra rank is slack the optimiser needs to
# reach an exact group homomorphism.
#
# A skewed-basin story predicts BIMODAL SEEDS, and Match-Query's 1-layer
# path-integrated arm is exactly that: 0.11 0.32 0.35 0.40 0.45 0.52 0.71 0.80
# (sd 0.220, LOOP_HEADROOM.md n=8). The loop already fixes it -- 8/8 seeds >=
# 0.77, sd 0.099 -- and LOOP_HEADROOM's own reading was that "the loop's
# contribution is mostly to the FLOOR". If r=4 does the same thing for 384
# parameters, the two are redundant and the loop is not buying what it appeared
# to.
#
# PRE-REGISTERED. r=4 should COMPRESS THE SEED VARIANCE MORE THAN IT LIFTS THE
# MEAN, i.e. min(r4) >> min(r2) with a smaller sd, mirroring the loop's profile.
# Reported as the primary quantity alongside the mean, because a mean-only read
# would miss it. Then:
#   - if the r4 x loop INTERACTION is negative and both mains are positive, they
#     are removing the same failure mode -> redundant, and 384 parameters is the
#     cheaper route.
#   - if the interaction is ~0 and both mains hold, they are independent fixes
#     and Looped_r4 should be the best arm measured on this task.
#   - if r=4 does nothing here, the skewed-basin account does not transfer off
#     the torus and the DXR optimisation half is torus-specific.
#
# PARAMETER-CLEAN. The loop is free on both rows (204,373 / 204,757 either way)
# and rank costs exactly +384 on both rows, so the interaction is
# parameter-matched, not a capacity contrast.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R=$REPO/runs/mq_rank
LOG=$REPO/mq_rank.log
mkdir -p "$R"

# Wait for the D x r batch. Count real interpreters only: `pgrep -f` matches any
# shell mentioning the pattern, including the tool call that launched this.
busy() { ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_variant/' | wc -l; }
echo "mq_rank queued $(date), waiting for .dxr_done" > "$LOG"
while [ ! -f "$REPO/.dxr_done" ] || [ "$(busy)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) dxr clear, starting" >> "$LOG"

# LOOP_HEADROOM's recipe exactly, so the arms sit beside its five.
EP=300; NB=48; BS=16; SZ=128; NOBS=16; TE=512; TQ=256; DM=128; NH=2
MAXPG=3
on_gpu() { ps -u "$USER" -o comm=,args= \
           | awk -v d="cuda:$1" '$1=="python3" && /mapformer\.train_match_query/ && index($0,d)' | wc -l; }

for SEED in 0 1 2 3 4 5 6 7; do
  for V in Vanilla Vanilla_r4 Looped Looped_r4; do
    OUT="$R/$V/s$SEED"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && continue
    while :; do
      N0=$(on_gpu 0); N1=$(on_gpu 1)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      sleep 30
    done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_match_query --variant "$V" --seed "$SEED" \
      --epochs $EP --n-batches $NB --batch-size $BS --size $SZ --n-obs $NOBS \
      --T-explore $TE --T-query $TQ --eval-query 256 512 --n-layers 1 \
      --d-model $DM --n-heads $NH --schedule cosine --fast-attn \
      --device "cuda:$G" --output-dir "$OUT" > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 20
  done
done
wait

# `wait` returns regardless of child success -- verify the artifacts.
NCK=$(find "$R" -name '*.pt' | wc -l)
echo "$(date +%H:%M) $NCK/32 checkpoints" >> "$LOG"
python3 -u -m mapformer.agg_mq_rank --runs-dir "$R" --tq 256 \
  --out "$REPO/MQ_RANK_2X2.md" >> "$LOG" 2>&1
if [ -f "$REPO/MQ_RANK_2X2.md" ]; then
  touch "$REPO/.mq_rank_done"; echo "$(date) DONE" >> "$LOG"
else
  echo "$(date) AGGREGATION FAILED -- no marker" >> "$LOG"
fi
