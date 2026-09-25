#!/usr/bin/env bash
# Warm-start stability test. Pre-registration: RANK_PROJ_PREREG.md.
# Scheduling, locking and the done marker come from lib_driver.sh (2026-09-24); jobs per GPU
# default to 2 (was 5): MAXPG=5 restores the old packing.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_proj.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"
drv_lock "$REPO/.run_rank_proj.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
echo "start $(date)" >> "$LOG"
P="$REPO/runs/rank_proj"; R="$REPO/runs/rank_proj_train"; mkdir -p "$R/p0"
SEEDS="0 1 2 3 4 5 6 7"
# 1. FROZEN: existence on all 8 seeds (eval only)
python3 -u -m mapformer.eval_noise_refine --runs-dir "$P" --variants Vanilla --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "Projected r=2 (from solved r=4), frozen" --out "$REPO/RANK_PROJ_FROZEN.md" >> "$LOG" 2>&1 || drv_fail frozen
# 2. TRAINABLE: the continuation's exact recipe from the projected weights
for SEED in $SEEDS; do
  OUT="$R/p0/Vanilla_s${SEED}"; mkdir -p "$OUT"
  if [ -f "$OUT/Vanilla.pt" ]; then
    # reuse only a checkpoint trained from THIS projection at THIS budget (hygiene patch 05)
    python3 - "$OUT/Vanilla.pt" "$P/p0/Vanilla_s${SEED}/Vanilla.pt" <<'PY' || drv_fail "stale checkpoint $OUT/Vanilla.pt"
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["config"]
sys.exit(0 if c.get("epochs") == 900 and c.get("n_steps") == 1024 and c.get("init_from") == sys.argv[2] else 1)
PY
    echo "skip s$SEED (done, config matches)" >> "$LOG"; continue
  fi
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) Vanilla(proj) s$SEED -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$R/Vanilla_s${SEED}.log" python3 -u -m mapformer.train_variant --variant Vanilla --seed "$SEED" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
    --init-from "$P/p0/Vanilla_s${SEED}/Vanilla.pt" --data-seed-offset 1 --save-full-state
done
drv_wait_dir "$R/"
CKPTS=""; for SEED in $SEEDS; do CKPTS="$CKPTS $R/p0/Vanilla_s${SEED}/Vanilla.pt"; done
drv_require $CKPTS || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla --noises 0.0 \
  --seeds $SEEDS --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "Projected r=2, trained 900 epochs (continuation recipe)" --out "$REPO/RANK_PROJ_TRAIN.md" >> "$LOG" 2>&1 || drv_fail train_eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla --seeds $SEEDS --lengths 1024 \
  --n-trials 100 --device cuda:1 --check-json "$REPO/RANK_PROJ_TRAIN.json" --out "$REPO/RANK_PROJ_TRAIN_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla --seeds $SEEDS --device cuda:1 \
  --out "$REPO/RANK_PROJ_TRAIN_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail geometry
# 3. the comparison needs the continuation's control arm -- and must not wait forever if that
# driver FAILED or ABORTED (its failures set no marker). Only its LATEST invocation counts.
E9C="$REPO/rank_matched_e900c.log"
until [ -f "$REPO/.rank_matched_e900c_done" ]; do
  [ -f "$E9C" ] && awk '/^start /{b=""} {b=b $0 "\n"} END{printf "%s", b}' "$E9C" |
      grep -qE "FAILED|ABORT|REFUSED|not all checkpoints" \
    && drv_fail "rank_matched_e900c driver failed -- see $E9C"
  sleep "${DRV_WAIT_POLL:-120}"
done
python3 -u -m mapformer.analyze_rank_proj > "$REPO/RANK_PROJ_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_proj_done" "$REPO/RANK_PROJ_FROZEN.md" "$REPO/RANK_PROJ_TRAIN.md" "$REPO/RANK_PROJ_TRAIN.json" \
  "$REPO/RANK_PROJ_TRAIN_STRATA.json" "$REPO/RANK_PROJ_TRAIN_GEOMETRY.md" "$REPO/RANK_PROJ_ANALYSIS.txt"
