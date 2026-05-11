#!/bin/bash
# Cognitive-map tier: long-T eval → sparse landmarks → multi-env, chained.
# Minimal 5-model set (RoPE / Vanilla / Level15 / Level15GSF_NoDrop / TEMFaithful)
# to keep compute manageable. Backfill of other baselines (LSTM / MambaLike /
# VanillaEM / NoDrop alone / GSF alone) is on the TODO list for paper submission.
set -u
REPO=/home/prashr/mapformer
echo "[$(date)] [cognitive-tier] starting"
echo "[$(date)] [cognitive-tier] step 1: long-T eval"
bash "$REPO/run_longt.sh"
echo "[$(date)] [cognitive-tier] step 2: sparse landmarks"
bash "$REPO/run_sparse_landmarks.sh"
echo "[$(date)] [cognitive-tier] step 3: multi-environment"
bash "$REPO/run_multienv.sh"
echo "[$(date)] [cognitive-tier] all done"
