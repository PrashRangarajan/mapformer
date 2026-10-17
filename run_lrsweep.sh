#!/usr/bin/env bash
# LRSWEEP_PREREG.md batch, seeds 1500-1507: ring +1/-1 (train_gauge, the GAUGE m-cell recipe: 32 cells, T 128, batch 128,
# 98 batches, cosine, 300 epochs, wd 0.05), observation-step-free SignedAO_r4 (SO) vs AbsAO_r4 (AO) at learning rates
# 1e-3 / 3e-3 / 1e-2 / 3e-2: 2 arms x 4 rates x 8 seeds = 64 runs, up to 6 trainers per GPU. Then the registered analysis
# (CPU). GPUs from DRV_GPUS. train_gauge.py is unchanged from ROBUST (lr and weight decay are existing flags).
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_lrsweep.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/lrsweep.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27); 4 per GPU: shared with another user (Amendment 1)
export DRV_GPUS="${DRV_GPUS:-0 1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_lrsweep.lock" || exit 1
echo "start $(date)" >> "$LOG"
# clean tree BEFORE any wait (a batch must run committed code)
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch analyze_lrsweep.py run_lrsweep.sh train_gauge.py LRSWEEP_PREREG.md > /dev/null 2>&1) ||
  { _drv_log "REFUSED: analyze_lrsweep.py / run_lrsweep.sh / train_gauge.py / LRSWEEP_PREREG.md not committed"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/lrsweep"; mkdir -p "$R"
SEEDS=(1500 1501 1502 1503 1504 1505 1506 1507)
RATES=(1e-3 3e-3 1e-2 3e-2)                       # an amendment extending the grid edits RATES and TAG together
declare -A TAG=([1e-3]=lr1e-03 [3e-3]=lr3e-03 [1e-2]=lr1e-02 [3e-2]=lr3e-02 [1e-1]=lr1e-01)  # = analyze_lrsweep.rate_tag
for L in "${RATES[@]}"; do
  [ "${TAG[$L]}" = "$(python3 -c "print(f'lr{float(\"$L\"):.0e}')")" ] || { _drv_log "ABORT: tag mismatch for $L"; exit 1; }
done
# md5 guard over EVERY mapformer module the batch imports (sys.modules after importing the trainer and the analysis;
# data_parallel is imported lazily inside train(), so it is imported explicitly), plus the driver files
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_gauge, mapformer.data_parallel, mapformer.analyze_lrsweep
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_lrsweep.sh)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
ring(){  # rate arm seed
  local L="$1" A="$2" S="$3" OUT="$R/${TAG[$1]}/$2_s$3" G
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $L $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $L $A s$S (running)"; return 0; }
  G=$(drv_wait_slot "$DRV_MAXPG"); mkdir -p "$OUT"; _drv_log "ring $L $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_gauge --arm "$A" --deltas=1,-1 --seed "$S" \
    --lr "$L" --weight-decay 0.05 --device "cuda:$G" --output-dir "$OUT"
}
# seed outer, rate, arm inner
for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in SO AO; do ring "$L" "$A" "$S"; done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in SO AO; do
  REQ+=("$R/${TAG[$L]}/${A}_s$S/eval.json" "$R/${TAG[$L]}/${A}_s$S/$A.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
NEXP=$(( 2 * 2 * ${#RATES[@]} * ${#SEEDS[@]} ))
[ "$NF" -eq "$NEXP" ] || drv_fail "expected $NEXP artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_lrsweep --root "$R" --seeds 1500-1507 \
  --rates "$(IFS=,; echo "${RATES[*]}")" > "$REPO/LRSWEEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.lrsweep_done" "$REPO/LRSWEEP_ANALYSIS.txt" "$REPO/LRSWEEP_VERDICTS.json"
