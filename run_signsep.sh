#!/usr/bin/env bash
# SIGNSEP_PREREG.md batch: seeds 2300-2307 x arms signed abs abs_wrap signed_frozen abs_wrap_past x lr 1e-3 3e-3
# (train_signsep.py), ring 32, +1/-1, 300 epochs, the common |.| start. 80 runs.
#   cd /home/prashr/mapformer && setsid nohup bash run_signsep.sh > /dev/null 2>&1 &
# GPU policy (user, 2026-10-23, host kalman): both GPUs carry another user's jobs; ours run beside them at <= 4 per GPU
# (DRV_MAXPG=4; lib_driver counts EVERY mapformer.train_* process of this user on a device, so other batches of ours
# share the same budget). Restrict with DRV_GPUS="1" to use one GPU.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/signsep.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=3000; DRV_SPACING=20      # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
drv_lock "$REPO/.run_signsep.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/signsep/p0"; mkdir -p "$R"
SEEDS=(2300 2301 2302 2303 2304 2305 2306 2307)
ARMS=(signed abs abs_wrap signed_frozen abs_wrap_past)
RATES=(1e-3 3e-3)
echo "start $(date)" >> "$LOG"
# clean tree BEFORE any launch or wait: tracked files unmodified AND every batch file committed
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch model_signsep.py train_signsep.py analyze_signsep.py construct_signsep.py \
   test_signsep.py run_signsep.sh SIGNSEP_PREREG.md > /dev/null 2>&1) ||
  { _drv_log "REFUSED: a SIGNSEP file is not committed"; exit 1; }
# directory tags must equal analyze_signsep.rate_tag, or the analysis reads nothing
for L in "${RATES[@]}"; do
  [ "lr$L" = "$(CUDA_VISIBLE_DEVICES="" python3 -c "from mapformer.analyze_signsep import rate_tag; print(rate_tag(float('$L')))" 2>/dev/null)" ] ||
    { _drv_log "ABORT: tag mismatch for $L"; exit 1; }
done
# md5 guard over every mapformer module the trainer and the analysis import (sys.modules), plus driver, lib, prereg;
# data_parallel is imported lazily inside train(), so explicitly
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_signsep, mapformer.data_parallel, mapformer.analyze_signsep
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_signsep.sh SIGNSEP_PREREG.md)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS max $DRV_MAXPG/GPU; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
# seed-major order: a partial batch is balanced across arms and rates
for S in "${SEEDS[@]}"; do for LR in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  OUT="$R/${A}_lr${LR}_s$S"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A lr$LR s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A lr$LR s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A lr$LR s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_signsep --arm "$A" --lr "$LR" \
    --seed "$S" --device "cuda:$G" --output-dir "$OUT"
done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for LR in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  REQ+=("$R/${A}_lr${LR}_s$S/eval.json" "$R/${A}_lr${LR}_s$S/${A}.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 160 ] || drv_fail "expected 160 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_signsep > "$REPO/SIGNSEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.signsep_done" "$REPO/SIGNSEP_ANALYSIS.txt" "$REPO/SIGNSEP_VERDICTS.json"
