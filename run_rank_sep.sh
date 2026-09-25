#!/usr/bin/env bash
# Rank separation. Pre-registration: RANK_SEP_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_sep.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"
DRV_MODULE_RE="mapformer[.]train_variant"      # own slots; a concurrent Dyck batch counts separately
drv_lock "$REPO/.run_rank_sep.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_sep"; RP="$REPO/runs/rank_sep_repro"; mkdir -p "$R/p0" "$RP/p0"
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" model.py model_rank.py model_rank_perhead.py train.py train_variant.py environment.py data_parallel.py || exit 1
launch(){ local ROOT=$1 V=$2 S=$3 G; OUT="$ROOT/p0/${V}_s${S}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
launch "$RP" Vanilla_r4mi 0
for S in 0 1 2 3 4 5 6 7; do launch "$R" Vanilla_r4mibd $S; launch "$R" Vanilla_r4ph $S; done
drv_wait_dir "$R/"; drv_wait_dir "$RP/"
REQ=("$RP/p0/Vanilla_r4mi_s0/Vanilla_r4mi.pt")
for S in 0 1 2 3 4 5 6 7; do for V in Vanilla_r4mibd Vanilla_r4ph; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla_r4mibd Vanilla_r4ph --noises 0.0 \
  --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank separation, trained and tested at T=1024" --out "$REPO/RANK_SEP.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_rank_strata --runs-dir "$R" --variants Vanilla_r4mibd Vanilla_r4ph --seeds 0 1 2 3 4 5 6 7 \
  --lengths 1024 --n-trials 100 --device cuda:0 --check-json "$REPO/RANK_SEP.json" --out "$REPO/RANK_SEP_STRATA.json" >> "$LOG" 2>&1 || drv_fail strata
python3 -u -m mapformer.probe_action_geometry --runs-dir "$R/p0" --arms Vanilla_r4mibd Vanilla_r4ph --seeds 0 1 2 3 4 5 6 7 \
  --space delta --device cuda:0 --out "$REPO/RANK_SEP_GEOMETRY.md" >> "$LOG" 2>&1 || drv_fail geometry
python3 -u -m mapformer.analyze_rank_sep > "$REPO/RANK_SEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_sep_done" "$REPO/RANK_SEP.md" "$REPO/RANK_SEP.json" "$REPO/RANK_SEP_STRATA.json" \
  "$REPO/RANK_SEP_GEOMETRY.md" "$REPO/RANK_SEP_ANALYSIS.txt"
