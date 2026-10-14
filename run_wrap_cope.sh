#!/usr/bin/env bash
# WRAP_COPE_PREREG.md batch: seeds 900-907 x 4 CoPE arms (train_ring_variant.py), ring, 300 epochs. GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_wrap_cope.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/wrap_cope.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27); shares slots with GENERALITY
export DRV_GPUS="${DRV_GPUS:-0 1}"
drv_lock "$REPO/.run_wrap_cope.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/wrap_cope/p0"; mkdir -p "$R"
SEEDS=(900 901 902 903 904 905 906 907); ARMS=(CoPEF CoPEF_Hi CoPEF_Lo CoPEF_LoS)
GUARD=(train_ring_variant.py model_mono_baselines.py model_sign.py train_cancel.py environment_cancel.py environment_nd.py model.py
       train_variant.py lie_groups.py train.py analyze_wrap_cope.py stats_core.py lib_driver.sh run_wrap_cope.sh
       data_parallel.py environment.py rescore_hook.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$REPO/runs/wrap_cope" "${GUARD[@]}" || exit 1
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do
  OUT="$R/${A}_s$S"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ring_variant --arm "$A" --p-plus 0.5 --seed "$S" --device "cuda:$G" \
    --output-dir "$OUT"
done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do REQ+=("$R/${A}_s$S/eval.json" "$R/${A}_s$S/$A.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 64 ] || drv_fail "expected 64 artifacts, found $NF"
drv_md5_guard "$REPO/runs/wrap_cope" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_wrap_cope --root "$R" --seeds 900-907 > "$REPO/WRAP_COPE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.wrap_cope_done" "$REPO/WRAP_COPE_ANALYSIS.txt" "$REPO/WRAP_COPE_VERDICTS.json"
