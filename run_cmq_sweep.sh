#!/usr/bin/env bash
# GUARD (2026-10-29 disk cleanup): the checkpoints of runs this driver writes or reads were
# deleted; it would retrain those runs or overwrite tracked results. Losses/configs survive in
# runs/<dir>/CKPT_META.jsonl. See docs/cleanup/DISK_TIERS_ANALYSIS.md section 4, D2_MANIFEST.md.
if [ "${MAPFORMER_ALLOW_DELETED_RERUN:-0}" != "1" ]; then
  echo "REFUSED: $(basename "$0"): checkpoints of its runs were deleted on 2026-10-29 (docs/cleanup/DISK_TIERS_ANALYSIS.md, section 4); relaunching would retrain them or overwrite tracked results. Set MAPFORMER_ALLOW_DELETED_RERUN=1 to run anyway (rule 30: reproduce archived batches from their pre-registration commit)." >&2
  exit 3
fi
# Compositional Match-Query sweep: 5 variants x 3 seeds = 15 runs, ALL launched
# concurrently and packed across both GPUs (round-robin). Each run is ~600K
# params / ~2 GB, so ~8 fit per 24 GB GPU; 32 cores absorb the CPU env-gen.
# Resumable: skips a run whose checkpoint already exists. Aggregates at the end.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO/.."                       # run modules from the package parent
LOGD="$REPO/runs/cmq_sweep"
mkdir -p "$LOGD"
LOG="$REPO/cmq_sweep.log"
echo "cmq sweep start $(date)" > "$LOG"

VARIANTS=(Vanilla Hourglass_k2 Hourglass_CoarseIdx PlainFlat PlainHourglass)
SEEDS=(0 1 2)

TE=512; TQ=256; EPOCHS=200; NB=48; BS=16

i=0
pids=()
for SEED in "${SEEDS[@]}"; do
  OUT="$LOGD/s${SEED}"
  for V in "${VARIANTS[@]}"; do
    GPU=$(( i % 2 ))
    if [ -f "$OUT/${V}_cmq.pt" ]; then
      echo "skip ${V} s${SEED} (exists)" >> "$LOG"; i=$((i+1)); continue
    fi
    echo "launch ${V} s${SEED} -> cuda:${GPU}" >> "$LOG"
    python3 -u -m mapformer.train_compositional_match_query \
        --variant "$V" --seed "$SEED" \
        --T-explore $TE --T-query $TQ --epochs $EPOCHS \
        --n-batches $NB --batch-size $BS --eval-query 256 512 \
        --device "cuda:${GPU}" --output-dir "$OUT" \
        > "$LOGD/${V}_s${SEED}.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
    sleep 2      # stagger CUDA-context creation to avoid an init thundering herd
  done
done

echo "all ${#pids[@]} runs launched; waiting" >> "$LOG"
wait "${pids[@]}"
echo "$(date) all runs done; aggregating" >> "$LOG"

python3 -u -m mapformer.agg_cmq --runs-dir "$LOGD" \
    --variants "${VARIANTS[@]}" \
    --out "$REPO/COMPOSITIONAL_MATCH_QUERY_RESULTS.md" >> "$LOG" 2>&1
touch "$REPO/.cmq_sweep_done"
echo "$(date) DONE" >> "$LOG"
