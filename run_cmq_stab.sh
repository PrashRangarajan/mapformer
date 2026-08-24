#!/usr/bin/env bash
# Stabilised Compositional Match-Query sweep: 5 variants x 6 seeds = 30 runs,
# WITH LR warmup (the fix for the bimodal training basin). Job-pool capped at
# MAXJ concurrent (~6/GPU -> ~18 GB, safe headroom; the earlier 8/GPU OOM'd).
# Resumable (skips existing checkpoints). Writes a SEPARATE results file so it
# does not clobber the no-warmup sweep.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."
LOGD="$REPO/runs/cmq_stab"; mkdir -p "$LOGD"
LOG="$REPO/cmq_stab.log"; echo "stab start $(date)" > "$LOG"

VARIANTS=(Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass)
SEEDS=(0 1 2 3 4 5)
MAXJ=12
TE=512; TQ=256; EPOCHS=200; NB=48; BS=16; WU=0.05

i=0
for SEED in "${SEEDS[@]}"; do
  OUT="$LOGD/s${SEED}"
  for V in "${VARIANTS[@]}"; do
    if [ -f "$OUT/${V}_cmq.pt" ]; then
      echo "skip ${V} s${SEED} (exists)" >> "$LOG"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
    GPU=$(( i % 2 ))
    echo "$(date +%H:%M) launch ${V} s${SEED} -> cuda:${GPU}" >> "$LOG"
    python3 -u -m mapformer.train_compositional_match_query \
        --variant "$V" --seed "$SEED" --T-explore $TE --T-query $TQ \
        --epochs $EPOCHS --n-batches $NB --batch-size $BS --warmup-frac $WU \
        --eval-query 256 512 --device "cuda:${GPU}" --output-dir "$OUT" \
        > "$LOGD/${V}_s${SEED}.log" 2>&1 &
    i=$((i+1)); sleep 2
  done
done
wait
echo "$(date) all done; aggregating" >> "$LOG"
python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" --variants "${VARIANTS[@]}" \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_STAB.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_stab_done"
echo "$(date) DONE" >> "$LOG"
