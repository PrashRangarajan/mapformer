#!/usr/bin/env bash
# Does PoPE's score rule rescue per-head rank 2 on the long-walk torus? Pre-registration: SCORE_RANK_PREREG.md.
# Recipe = run_rank_mi.sh's exactly (T=1024, 900 epochs, B16, lr 1e-3, cosine, 1 layer, 2 heads, d 128, data-workers 3,
# explicit attention path). Arms: Vanilla, MapPoPE-Pair (rank 2, seeds 10-21); Vanilla_r4mi, MapPoPE-Pair_r4mi
# (rank 4, seeds 10-17). Seed outer, variant inner. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash run_score_rank.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/score_rank.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-20}"   # 2/GPU: same throughput as 4/GPU in the pilot
drv_lock "$REPO/.run_score_rank.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/score_rank"; mkdir -p "$R/p0"
R2ARMS="Vanilla MapPoPE-Pair"; R4ARMS="Vanilla_r4mi MapPoPE-Pair_r4mi"
R2SEEDS="$(seq -s ' ' 10 21)"; R4SEEDS="$(seq -s ' ' 10 17)"
GUARD=(model.py model_rank_perhead.py model_pope.py model_pope_pair.py model_pope_pair_mi.py train.py train_variant.py
       train_score_rank.py eval_score_rank.py eval_noise_refine.py eval_rank_strata.py rescore_hook.py ckpt_guard.py
       environment.py data_parallel.py analyze_score_rank.py stats_core.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
launch(){ local V=$1 S=$2 G OUT="$R/p0/${1}_s${2}"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $V s$S" >> "$LOG"; return; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=4 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_score_rank --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state; }
for S in $R2SEEDS; do
  for V in $R2ARMS; do launch "$V" "$S"; done
  [ "$S" -le 17 ] && for V in $R4ARMS; do launch "$V" "$S"; done
done
drv_wait_dir "$R/"
REQ=()
for S in $R2SEEDS; do for V in $R2ARMS; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
for S in $R4SEEDS; do for V in $R4ARMS; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
[ "${#REQ[@]}" -eq 40 ] || drv_fail "expected 40 checkpoints, listed ${#REQ[@]}"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
EV(){ python3 -u -m mapformer.eval_score_rank "$@"; }
for G in R2 R4; do
  if [ "$G" = R2 ]; then ARMS=$R2ARMS; SEEDS=$R2SEEDS; else ARMS=$R4ARMS; SEEDS=$R4SEEDS; fi
  EV mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 --seeds $SEEDS \
    --lengths 512 1024 2048 --n-trials 100 --device cuda:0 --out "$REPO/SCORE_RANK_$G.md" \
    --title "Score rule x rank on the long-walk torus, trained and tested at T=1024 ($G)" >> "$LOG" 2>&1 || drv_fail "eval $G"
  EV mapformer.eval_rank_strata --runs-dir "$R" --variants $ARMS --seeds $SEEDS --lengths 1024 --n-trials 100 \
    --device cuda:0 --check-json "$REPO/SCORE_RANK_$G.json" --out "$REPO/SCORE_RANK_STRATA_$G.json" >> "$LOG" 2>&1 || drv_fail "strata $G"
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_score_rank mapformer.eval_noise_refine --runs-dir "$R" \
    --variants $ARMS --noises 0.0 --seeds $SEEDS --lengths 1024 --n-trials 100 --device cuda:0 \
    --out "$REPO/SCORE_RANK_RESCORE_$G.md" --title "SCORE_RANK dropout-scale re-score ($G)" >> "$LOG" 2>&1 || drv_fail "rescore $G"
done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_score_rank > "$REPO/SCORE_RANK_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.score_rank_done" "$REPO/SCORE_RANK_R2.json" "$REPO/SCORE_RANK_R4.json" "$REPO/SCORE_RANK_STRATA_R2.json" \
  "$REPO/SCORE_RANK_STRATA_R4.json" "$REPO/SCORE_RANK_RESCORE_R2.json" "$REPO/SCORE_RANK_RESCORE_R4.json" \
  "$REPO/SCORE_RANK_ANALYSIS.txt"
