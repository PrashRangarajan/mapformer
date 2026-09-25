#!/usr/bin/env bash
# Rank at matched length. Pre-registration: RANK_MATCHED_PREREG.md (Amendments 1-2).
#   300-epoch batch:        bash run_rank_matched.sh
#   900-epoch pilot:        EPOCHS=900 TAG=_e900 SEEDS="0 1" PILOT=1 bash run_rank_matched.sh
#   900-epoch full batch:   EPOCHS=900 TAG=_e900 bash run_rank_matched.sh   (all 8 seeds; the
#                           pilot's seeds 0-1 are reused, verified by epochs and code md5)
#   continuation (Amdt 3):  EPOCHS=900 TAG=_e900c INIT_TAG=_e900 bash run_rank_matched.sh
#                           (each run starts from runs/rank_matched_e900's weights: a warm
#                           restart of the same recipe on a fresh data stream)
# Scheduling, locking, the md5 guard and the done marker come from lib_driver.sh (2026-09-24).
# Jobs per GPU default to 2 (was 5; efficiency audit #4): MAXPG=5 restores the old packing.
# NOTE: the 2026-09-24 efficiency patches changed the md5 of environment.py, model.py, train.py
# and train_variant.py (verified bit-identical on the default path, docs/audits/2026-09-24/
# applied/), so re-invoking a series whose code_md5.txt predates them ABORTS by design.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
EPOCHS="${EPOCHS:-300}"; TAG="${TAG:-}"; PILOT="${PILOT:-0}"; INIT_TAG="${INIT_TAG:-}"
R="$REPO/runs/rank_matched${TAG}"; mkdir -p "$R/p0"
LOG="$REPO/rank_matched${TAG}.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"
drv_lock "$REPO/.run_rank_matched.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
echo "start $(date) epochs=$EPOCHS pilot=$PILOT" >> "$LOG"
# markers from an earlier invocation (e.g. the pilot) must not read as this run being done
rm -f "$R/.train_done" "$REPO/.rank_matched${TAG}_done"
# the training code must be the code any reused checkpoint was trained with
MODS="model.py model_rank.py train.py train_variant.py environment.py data_parallel.py"
drv_md5_guard "$R" $MODS || exit 1
ARMS="Vanilla Vanilla_r4"; SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
# a checkpoint is reusable only if it was trained at the requested budget and length
matches(){ python3 - "$1" "$EPOCHS" "$2" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
ok = c.get("epochs") == int(sys.argv[2]) and c.get("n_steps") == 1024 and c.get("batch_size") == 16
ok = ok and (c.get("init_from") or "") == sys.argv[3]
sys.exit(0 if ok else 1)
PY
}
init_for(){ [ -n "$INIT_TAG" ] && echo "$REPO/runs/rank_matched${INIT_TAG}/p0/$1_s$2/$1.pt" || echo ""; }
for SEED in $SEEDS; do
  for V in $ARMS; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    if [ -f "$OUT/${V}.pt" ]; then
      matches "$OUT/${V}.pt" "$(init_for $V $SEED)" || { echo "ABORT: $OUT/${V}.pt exists but not at epochs=$EPOCHS n_steps=1024 bs=16 init=$(init_for $V $SEED)" >> "$LOG"; exit 1; }
      echo "skip $V s$SEED (done, config matches)" >> "$LOG"; continue
    fi
    INIT=$(init_for $V $SEED); EXTRA=""
    if [ -n "$INIT" ]; then
      [ -f "$INIT" ] || { echo "ABORT: init checkpoint missing $INIT" >> "$LOG"; exit 1; }
      EXTRA="--init-from $INIT --data-seed-offset 1 --save-full-state"
    fi
    G=$(drv_wait_slot)
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    OMP_NUM_THREADS=4 drv_launch "$R/${V}_s${SEED}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs "$EPOCHS" --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" $EXTRA
  done
done
drv_wait_dir "$R/"
CKPTS=""; for SEED in $SEEDS; do for V in $ARMS; do CKPTS="$CKPTS $R/p0/${V}_s${SEED}/${V}.pt"; done; done
drv_require $CKPTS || { echo "$(date +%H:%M) not all checkpoints present -- evaluation NOT run" >> "$LOG"; exit 1; }
echo "$(date +%H:%M) missing=0" >> "$LOG"
touch "$R/.train_done"
if [ "$PILOT" = 1 ]; then
  # a pilot reads TRAINING LOSS ONLY (Amendment 1-2)
  python3 -u -m mapformer.analyze_rank_matched --tag "$TAG" --seeds $SEEDS --classify-only >> "$LOG" 2>&1 \
    || drv_fail "classification"
  drv_done "$REPO/.rank_matched${TAG}_done"; echo "$(date) PILOT DONE (no eval)" >> "$LOG"; exit 0
fi
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank at matched length: trained AND tested at T=1024 (epochs=$EPOCHS)" \
  --out "$REPO/RANK_MATCHED${TAG}.md" >> "$LOG" 2>&1 || drv_fail eval_noise_refine
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:0 \
  --check-json "$REPO/RANK_MATCHED${TAG}.json" --out "$REPO/RANK_MATCHED${TAG}_STRATA.json" >> "$LOG" 2>&1 || drv_fail eval_rank_strata
[ -f "$REPO/RANK_SWEEP_STRATA.json" ] || python3 -u -m mapformer.eval_rank_strata --runs-dir "$REPO/runs/rank_sweep" --variants $ARMS --seeds $SEEDS \
  --lengths 1024 2048 --n-trials 100 --device cuda:1 \
  --check-json "$REPO/RANK_SWEEP.json" --out "$REPO/RANK_SWEEP_STRATA.json" >> "$LOG" 2>&1 || drv_fail "old strata"
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms $ARMS --seeds $SEEDS --device cuda:0 \
  --out "$REPO/RANK_MATCHED${TAG}_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail probe_action_geometry
python3 -u -m mapformer.analyze_rank_matched --tag "$TAG" --seeds $SEEDS > "$REPO/RANK_MATCHED${TAG}_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_matched${TAG}_done" "$REPO/RANK_MATCHED${TAG}.md" "$REPO/RANK_MATCHED${TAG}.json" \
  "$REPO/RANK_MATCHED${TAG}_STRATA.json" "$REPO/RANK_MATCHED${TAG}_GEOMETRY.md" "$REPO/RANK_MATCHED${TAG}_ANALYSIS.txt"
