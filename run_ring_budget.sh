#!/usr/bin/env bash
# RING_BUDGET_PREREG.md + Amendment 1: Abs_r4 on the ring, p_plus 0.5, seeds 40-47, at 900 epochs AND a 300-epoch twin; GPUs from DRV_GPUS (default 0).
#   cd /home/prashr/mapformer && DRV_GPUS=0 setsid nohup bash run_ring_budget.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/ring_budget.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15     # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0}"
drv_lock "$REPO/.run_ring_budget.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/ring_budget/p0"; mkdir -p "$R"
GUARD=(train_ring_dial.py train_cancel.py environment_cancel.py environment_nd.py environment.py data_parallel.py train_variant.py
       model.py model_sign.py train.py analyze_ring_budget.py analyze_ring_dial.py analyze_rank_tests.py rescore_hook.py
       stats_core.py lib_driver.sh run_ring_budget.sh lie_groups.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$REPO/runs/ring_budget" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in 40 41 42 43 44 45 46 47; do for E in 900 300; do
  OUT="$R/Abs_r4_p0.5_e${E}_s${S}"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip e$E s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip e$E s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "Abs_r4 p0.5 e$E s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ring_dial --arm Abs_r4 --p-plus 0.5 --seed "$S" \
    --epochs "$E" --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_ring_dial/ && /runs\/ring_budget\/p0/' | grep -q .; do sleep 60; done
REQ=(); for S in 40 41 42 43 44 45 46 47; do for E in 900 300; do REQ+=("$R/Abs_r4_p0.5_e${E}_s${S}/eval.json" "$R/Abs_r4_p0.5_e${E}_s${S}/Abs_r4.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
drv_md5_guard "$REPO/runs/ring_budget" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_ring_budget > "$REPO/RING_BUDGET_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.ring_budget_done" "$REPO/RING_BUDGET_ANALYSIS.txt" "$REPO/RING_BUDGET_VERDICTS.json"
