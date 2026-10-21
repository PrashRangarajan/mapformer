#!/usr/bin/env bash
# SIGNROUTE_PREREG.md batch (G-2 Part A + fixed-rate breadth): seeds 1800-1807 x arms
# signed abs hybrid abs_tied exp softplus x lr 1e-3 3e-3 (train_signroute.py), ring 32, 300 epochs.
# 96 runs. One GPU by the user's instruction (2026-10-20); GPU 1 by default.
#   cd /home/prashr/mapformer && setsid nohup bash run_signroute.sh > /dev/null 2>&1 &
# DRV_MAXPG=1: measured on this host 2026-10-20, three concurrent ring runs take 3.1 s/epoch
# against 0.8 s alone, i.e. packing LOWERS throughput here (rule 24: place by measured occupancy).
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/signroute.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=1; DRV_MINFREE=3000; DRV_SPACING=5      # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-1}"
drv_lock "$REPO/.run_signroute.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/signroute/p0"; mkdir -p "$R"
SEEDS=(1800 1801 1802 1803 1804 1805 1806 1807)
ARMS=(signed abs hybrid abs_tied exp softplus)
RATES=(1e-3 3e-3)
echo "start $(date)" >> "$LOG"
# clean tree BEFORE anything else: tracked files unmodified (Amendment 1 finding 3) AND every batch file
# committed -- `--untracked-files=no` alone passes a batch whose own code was never committed
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch train_signroute.py model_signroute.py analyze_signroute.py \
   run_signroute.sh test_signroute.py construct_signroute.py SIGNROUTE_PREREG.md > /dev/null 2>&1) ||
  { _drv_log "REFUSED: a SIGNROUTE file is not committed"; exit 1; }
# directory tags must equal analyze_signroute.rate_tag, or the analysis reads nothing
for L in "${RATES[@]}"; do
  [ "lr$L" = "$(CUDA_VISIBLE_DEVICES="" python3 -c "from mapformer.analyze_signroute import rate_tag; print(rate_tag(float('$L')))" 2>/dev/null)" ] ||
    { _drv_log "ABORT: tag mismatch for $L"; exit 1; }
done
# md5 guard over EVERY mapformer module the batch imports (Amendment 1 finding 4; as run_lrsweep.sh): sys.modules
# after importing the trainer and the analysis; data_parallel is imported lazily inside train(), so explicitly
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_signroute, mapformer.data_parallel, mapformer.analyze_signroute
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_signroute.sh SIGNROUTE_PREREG.md)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
# seed-major order: a partial batch is balanced across arms and rates rather than arm-complete
for S in "${SEEDS[@]}"; do for LR in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  OUT="$R/${A}_lr${LR}_s$S"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A lr$LR s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A lr$LR s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A lr$LR s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_signroute --arm "$A" --lr "$LR" \
    --seed "$S" --device "cuda:$G" --output-dir "$OUT"
done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for LR in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  REQ+=("$R/${A}_lr${LR}_s$S/eval.json" "$R/${A}_lr${LR}_s$S/${A}.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 192 ] || drv_fail "expected 192 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_signroute > "$REPO/SIGNROUTE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.signroute_done" "$REPO/SIGNROUTE_ANALYSIS.txt" "$REPO/SIGNROUTE_VERDICTS.json"
