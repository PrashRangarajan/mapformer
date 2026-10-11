#!/usr/bin/env bash
# RANK_TESTS_PREREG.md + Amendment 1: experiment 1 (ring: R1, R2, R1DF, R1AO, R2S; seeds 20-27; 300 epochs) and
# experiment 2 (rank escape: E lifted rank 4, C rank 2 continued, F fresh rank-4 bottleneck on the trained network;
# seeds 0-7; 600 epochs).
# Launch (only when no other user's jobs hold the GPUs):
#   cd /home/prashr/mapformer && setsid nohup bash run_rank_tests.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_tests.log"
source "$REPO/lib_driver.sh"
ESC_EPOCHS=600                      # Amendment 1 (was 300 with a pilot rule)
DRV_SPACING=20; DRV_MINFREE=5500    # assigned after sourcing (rule 27)
drv_lock "$REPO/.run_rank_tests.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/.."
R="$REPO/runs/rank_tests"; mkdir -p "$R/ring/p0" "$R/escape"
GUARD=(model.py model_rank_perhead.py model_ring_rank.py ring_rank_readouts.py train.py train_variant.py train_cancel.py
       train_ring_rank.py environment.py environment_nd.py environment_cancel.py data_parallel.py eval_noise_refine.py
       eval_rank_strata.py rescore_hook.py ckpt_guard.py analyze_rank_tests.py analyze_score_rank.py stats_core.py
       build_rank_lift.py lib_driver.sh run_rank_tests.sh docs/audits/2026-10-11/rank_tests_secondary.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); ESC_EPOCHS $ESC_EPOCHS"

# ---------------------------------------------------------------- experiment 1: ring
DRV_MAXPG=4
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }   # audit B4
for S in 20 21 22 23 24 25 26 27; do for A in R1 R2 R1DF R1AO R2S; do
  OUT="$R/ring/p0/${A}_s${S}"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A s$S (already running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "ring $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ring_rank --arm "$A" --seed "$S" \
    --device "cuda:$G" --output-dir "$OUT"
done; done

# ---------------------------------------------------------------- experiment 2: escape
python3 -m mapformer.build_rank_lift --out "$R/escape/lifted" >> "$LOG" 2>&1 || drv_fail "lift"
python3 -m mapformer.build_rank_lift --fresh --out "$R/escape/fresh" >> "$LOG" 2>&1 || drv_fail "fresh"
DRV_MAXPG=2
esc(){ local ARM=$1 V=$2 S=$3 INIT=$4 OUT="$R/escape/$1/p0/${2}_s${3}" G
  [ -f "$OUT/${V}.pt" ] && { _drv_log "skip $ARM s$S (done)"; return; }
  running "$OUT" && { _drv_log "skip $ARM s$S (already running)"; return; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "escape $ARM ($V) s$S -> cuda:$G"
  OMP_NUM_THREADS=4 drv_launch "$R/escape/${ARM}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs "$ESC_EPOCHS" --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --data-seed-offset 1 --device "cuda:$G" --output-dir "$OUT" $INIT; }
for S in 0 1 2 3 4 5 6 7; do
  esc E Vanilla_r4mi "$S" "--init-from $R/escape/lifted/p0/Vanilla_r4mi_s${S}/Vanilla_r4mi.pt"
  esc C Vanilla "$S" "--init-from $REPO/runs/rank_mi/p0/Vanilla_s${S}/Vanilla.pt"
  esc F Vanilla_r4mi "$S" "--init-from $R/escape/fresh/p0/Vanilla_r4mi_s${S}/Vanilla_r4mi.pt"
done
drv_wait_dir "$R/"
REQ=()
for S in 20 21 22 23 24 25 26 27; do for A in R1 R2 R1DF R1AO R2S; do REQ+=("$R/ring/p0/${A}_s${S}/${A}.pt" "$R/ring/p0/${A}_s${S}/eval.json"); done; done
for S in 0 1 2 3 4 5 6 7; do REQ+=("$R/escape/E/p0/Vanilla_r4mi_s${S}/Vanilla_r4mi.pt" "$R/escape/C/p0/Vanilla_s${S}/Vanilla.pt" \
  "$R/escape/F/p0/Vanilla_r4mi_s${S}/Vanilla_r4mi.pt"); done
[ "${#REQ[@]}" -eq 104 ] || drv_fail "expected 104 artifacts, listed ${#REQ[@]}"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
for ARM in E C F; do
  V=Vanilla_r4mi; [ "$ARM" = C ] && V=Vanilla
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/escape/$ARM" --variants "$V" --noises 0.0 --seeds 0 1 2 3 4 5 6 7 \
    --lengths 512 1024 2048 --n-trials 100 --device cuda:0 --title "Rank escape arm $ARM" \
    --out "$REPO/RANK_TESTS_ESCAPE_${ARM}.md" >> "$LOG" 2>&1 || drv_fail "eval $ARM"
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_noise_refine --runs-dir "$R/escape/$ARM" --variants "$V" \
    --noises 0.0 --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 2048 --n-trials 100 --device cuda:0 --title "Rank escape arm $ARM, re-scored" \
    --out "$REPO/RANK_TESTS_ESCAPE_${ARM}_RESCORED.md" >> "$LOG" 2>&1 || drv_fail "eval rescored $ARM"   # audit B2
  python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/escape/$ARM" --variants "$V" --seeds 0 1 2 3 4 5 6 7 --lengths 1024 \
    --n-trials 100 --device cuda:0 --out "$REPO/RANK_TESTS_ESCAPE_${ARM}_STRATA.json" >> "$LOG" 2>&1 || drv_fail "strata $ARM"
done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
python3 -u -m mapformer.analyze_rank_tests > "$REPO/RANK_TESTS_ANALYSIS.txt" 2>&1 || drv_fail analyze
PYTHONPATH="$REPO/.." CUDA_VISIBLE_DEVICES="" python3 -u "$REPO/docs/audits/2026-10-11/rank_tests_secondary.py" \
  > "$REPO/docs/audits/2026-10-11/rank_tests_secondary_out.txt" 2>&1 || _drv_log "secondaries failed (no verdict depends on them)"
drv_done "$REPO/.rank_tests_done" "$REPO/RANK_TESTS_ANALYSIS.txt" "$REPO/RANK_TESTS_VERDICTS.json"
