#!/usr/bin/env bash
# Sign ablation at matched length. Pre-registration: SIGN_MATCHED_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/sign_matched.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
DRV_MODULE_RE="mapformer[.]train_variant"
drv_lock "$REPO/.run_sign_matched.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/sign_matched"; mkdir -p "$R/p0"
ARMS="Signed_r4 Abs_r4 Pos_r4 RoPE"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_sign.py train.py train_variant.py environment.py data_parallel.py || exit 1
launch(){ local V=$1 S=$2 G; OUT="$R/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
for S in 0 1 2 3 4 5 6 7; do for V in $ARMS; do launch "$V" $S; done; done   # seed outer, arm inner
drv_wait_dir "$R/"
REQ=(); for S in 0 1 2 3 4 5 6 7; do for V in $ARMS; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Sign ablation, trained and tested at T=1024" --out "$REPO/SIGN_MATCHED.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants $ARMS --seeds 0 1 2 3 4 5 6 7 \
  --lengths 1024 --n-trials 100 --device cuda:0 --check-json "$REPO/SIGN_MATCHED.json" \
  --out "$REPO/SIGN_MATCHED_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_sign --runs-dir "$R" --variants Abs_r4 Pos_r4 --seeds 0 1 2 3 4 5 6 7 \
  --device cuda:0 --out "$REPO/SIGN_MATCHED_PROBE.md" >> "$LOG" 2>&1 || drv_fail probe
python3 -u -m mapformer.analyze_sign_matched > "$REPO/SIGN_MATCHED_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.sign_matched_done" "$REPO/SIGN_MATCHED.md" "$REPO/SIGN_MATCHED.json" "$REPO/SIGN_MATCHED_STRATA.json" \
  "$REPO/SIGN_MATCHED_PROBE.md" "$REPO/SIGN_MATCHED_ANALYSIS.txt"
