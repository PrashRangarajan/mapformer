#!/usr/bin/env bash
# H1: does a search aid recover rank 2? Pre-registration: LOOP_RANK_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/loop_rank.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
DRV_MODULE_RE="mapformer[.]train_variant"
drv_lock "$REPO/.run_loop_rank.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/loop_rank"; RP="$REPO/runs/loop_rank_repro"; mkdir -p "$R/p0" "$RP/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_looped.py train.py train_variant.py environment.py data_parallel.py || exit 1
launch(){ local ROOT=$1 V=$2 S=$3 NL=$4 G; OUT="$ROOT/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S (n_layers=$NL) -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers "$NL" --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
launch "$RP" Vanilla 0 1
for S in 0 1 2 3 4 5 6 7; do launch "$R" Looped $S 1; launch "$R" Vanilla_L4 $S 4; done
drv_wait_dir "$R/"; drv_wait_dir "$RP/"
REQ=("$RP/p0/Vanilla_s0/Vanilla.pt")
for S in 0 1 2 3 4 5 6 7; do for V in Looped Vanilla_L4; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Looped Vanilla_L4 --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "H1: loop and depth against rank 2, trained and tested at T=1024" \
  --out "$REPO/LOOP_RANK.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Looped Vanilla_L4 --seeds 0 1 2 3 4 5 6 7 \
  --lengths 1024 --n-trials 100 --device cuda:0 --check-json "$REPO/LOOP_RANK.json" \
  --out "$REPO/LOOP_RANK_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Looped Vanilla_L4 --seeds 0 1 2 3 4 5 6 7 \
  --space delta --device cuda:0 --out "$REPO/LOOP_RANK_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail geometry
python3 -u -m mapformer.analyze_loop_rank > "$REPO/LOOP_RANK_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.loop_rank_done" "$REPO/LOOP_RANK.md" "$REPO/LOOP_RANK.json" "$REPO/LOOP_RANK_STRATA.json" \
  "$REPO/LOOP_RANK_GEOMETRY.md" "$REPO/LOOP_RANK_ANALYSIS.txt"
