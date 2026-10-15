#!/usr/bin/env bash
# GAUGE_PREREG.md batch: seeds 1300-1307 x cells SO_m AO_m SO_p AO_p RoPE_p (train_gauge.py), ring, 300 epochs.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_gauge.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/gauge.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
drv_lock "$REPO/.run_gauge.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/gauge/p0"; mkdir -p "$R"
SEEDS=(1300 1301 1302 1303 1304 1305 1306 1307)
GUARD=(train_gauge.py environment_steps.py environment_cancel.py environment_nd.py model_wrap_exist.py model_sign.py model.py
       model_baseline_rope.py train_variant.py lie_groups.py train.py train_cancel.py data_parallel.py environment.py
       analyze_gauge.py stats_core.py lib_driver.sh run_gauge.sh)
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in "${SEEDS[@]}"; do for C in SO_m AO_m SO_p AO_p RoPE_p; do
  A="${C%_*}"; D=$([ "${C#*_}" = m ] && echo "1,-1" || echo "1,2"); OUT="$R/${C}_s$S"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $C s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $C s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$C s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_gauge --arm "$A" --deltas="$D" --seed "$S" \
    --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for C in SO_m AO_m SO_p AO_p RoPE_p; do REQ+=("$R/${C}_s$S/eval.json" "$R/${C}_s$S/${C%_*}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 80 ] || drv_fail "expected 80 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_gauge > "$REPO/GAUGE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.gauge_done" "$REPO/GAUGE_ANALYSIS.txt" "$REPO/GAUGE_VERDICTS.json"
