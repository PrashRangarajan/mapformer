#!/usr/bin/env bash
# DELTA_FREE_PREREG.md batch: phase {index, path} x PoPE delta {pinned, free} + MapPoPE with delta frozen at U(-2pi, 0).
# Paper torus, T = 128, r = 2, PAPER2X2's recipe exactly (run_paper2x2.sh: 300 epochs x 98 batches x 128, cosine, lr 1e-3,
# 1 layer, 2 heads, d 128, no landmarks, serial data). 5 arms x 8 seeds (1600-1607) = 40 runs, <= DRV_MAXPG (4) trainers
# per GPU on DRV_GPUS (required: the GPUs are shared). Then eval_noise_refine (PAPER2X2's eval: held-out map, T = 128 /
# 512 / 1024, 100 trials) on the first GPU of DRV_GPUS, then the registered analysis (CPU).
#   batch:  cd /home/prashr/mapformer && DRV_GPUS="1" setsid nohup bash run_delta_free.sh > /dev/null 2>&1 &
#   pilot:  MODE=pilot  (seed 1690, outside the batch, never pooled; runs/delta_free_pilot; no clean-tree requirement)
#   smoke:  MODE=smoke SMOKE_DIR=/abs/dir  (CPU, tiny recipe, seeds 1680-1687; exercises launch, eval and analysis)
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-batch}"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=3000; DRV_SPACING=15         # assigned after sourcing (rule 27); 4 per GPU: shared with another user
ARMS=(PoPE-Flat PoPE-FreeDelta MapPoPE-Flat MapPoPE-FreeDelta MapPoPE-FrozenDelta)
RECIPE=(--epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128
        --n-landmarks 0 --schedule cosine --lr 1e-3)
EVAL_TRIALS=100
case "$MODE" in
  batch) R="$REPO/runs/delta_free"; SEEDS=(1600 1601 1602 1603 1604 1605 1606 1607); LOG="$REPO/delta_free.log"
         RAW="$REPO/_DELTA_FREE_RAW.md"; MARK="$REPO/.delta_free_done" ;;
  pilot) R="$REPO/runs/delta_free_pilot"; SEEDS=(1690); LOG="$R/driver.log"; RAW="$R/_PILOT_RAW.md"; MARK="$R/.pilot_done" ;;
  smoke) R="${SMOKE_DIR:?SMOKE_DIR must be set}"; SEEDS=(1680 1681 1682 1683 1684 1685 1686 1687); LOG="$R/driver.log"
         RAW="$R/_SMOKE_RAW.md"; MARK="$R/.smoke_done"; DRV_SPACING=0; EVAL_TRIALS=3
         RECIPE=(--epochs 2 --n-batches 2 --batch-size 8 --n-steps 32 --n-layers 1 --n-heads 2 --d-model 128
                 --n-landmarks 0 --schedule cosine --lr 1e-3) ;;
  *) echo "MODE must be batch, pilot or smoke"; exit 1 ;;
esac
mkdir -p "$R/p0"
if [ "$MODE" != smoke ]; then
  [ -n "${DRV_GPUS:-}" ] || { _drv_log "REFUSED: set DRV_GPUS (the GPUs are shared; name the one(s) to use)"; exit 1; }
  export DRV_GPUS
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_delta_free.lock" || exit 1          # one lock for every mode: pilot and batch never overlap
_drv_log "start mode=$MODE seeds=${SEEDS[*]} gpus=${DRV_GPUS:-cpu}"
if [ "$MODE" = batch ]; then                             # clean tree BEFORE any wait (a batch must run committed code)
  (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
  (cd "$REPO" && git ls-files --error-unmatch analyze_delta_free.py run_delta_free.sh model_pope_delta.py DELTA_FREE_PREREG.md docs/audits/2026-10-20/delta_free_floors_out.txt > /dev/null 2>&1) ||
    { _drv_log "REFUSED: analyze_delta_free.py / run_delta_free.sh / model_pope_delta.py / DELTA_FREE_PREREG.md not committed"; exit 1; }
fi
cd "$REPO/.."
# md5 guard over EVERY mapformer module the trainer, the evaluator and the analysis import (sys.modules), plus the driver
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.analyze_delta_free
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
printf '%s\n' "${GUARD[@]}" | grep -qx model_pope_delta.py || { _drv_log "ABORT: model_pope_delta.py not in the guard"; exit 1; }
GUARD+=(lib_driver.sh run_delta_free.sh docs/audits/2026-10-20/delta_free_floors_out.txt)   # floors: Amendment 1
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) ($(cd "$REPO" && git status --porcelain --untracked-files=no | wc -l) tracked files modified); md5 guard over ${#GUARD[@]} files"

running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
train(){  # arm seed
  local V="$1" S="$2" OUT="$R/p0/$1_s$2" G DEV
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $V s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $V s$S (running)"; return 0; }
  if [ "$MODE" = smoke ]; then DEV=cpu; else G=$(drv_wait_slot "$DRV_MAXPG"); DEV="cuda:$G"; fi
  mkdir -p "$OUT"; _drv_log "train $V s$S -> $DEV"
  if [ "$MODE" = smoke ]; then        # foreground, tiny
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" "${RECIPE[@]}" \
      --device cpu --output-dir "$OUT" > "$R/${V}_s${S}.log" 2>&1
  else
    OMP_NUM_THREADS=2 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
      "${RECIPE[@]}" --device "$DEV" --output-dir "$OUT"
  fi
}
# seed outer, arm inner (feedback_seed_ordering)
for S in "${SEEDS[@]}"; do for V in "${ARMS[@]}"; do train "$V" "$S"; done; done
sleep 5
DRV_WAIT_POLL=60 drv_wait_dir "$R/p0/"
REQ=(); for S in "${SEEDS[@]}"; do for V in "${ARMS[@]}"; do REQ+=("$R/p0/${V}_s$S/$V.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
NEXP=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
[ "${#REQ[@]}" -eq "$NEXP" ] || drv_fail "expected $NEXP checkpoints, listed ${#REQ[@]}"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"

# evaluation (PAPER2X2's, unchanged): first GPU of DRV_GPUS, or CPU for the smoke
if [ "$MODE" = smoke ]; then EDEV=cpu; export CUDA_VISIBLE_DEVICES=""; else EDEV="cuda:${DRV_GPUS%% *}"; fi
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants "${ARMS[@]}" --noises 0.0 \
  --seeds "${SEEDS[@]}" --lengths 128 512 1024 --n-trials "$EVAL_TRIALS" --device "$EDEV" \
  --out "$RAW" --title "DELTA_FREE raw evaluation ($MODE)" > "$R/eval.log" 2>&1 || drv_fail "eval"
[ -s "${RAW%.md}.json" ] || drv_fail "no eval json"

if [ "$MODE" = pilot ]; then
  drv_done "$MARK" "${RAW%.md}.json"; exit 0
fi
if [ "$MODE" = smoke ]; then
  CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_delta_free --root "$R" --raw "${RAW%.md}.json" \
    --seeds 1680-1687 --floors none --probe-trials "$EVAL_TRIALS" --recipe smoke --json-out "$R/SMOKE_VERDICTS.json" > "$R/SMOKE_ANALYSIS.txt" 2>&1 || drv_fail analyze
  drv_done "$MARK" "$R/SMOKE_ANALYSIS.txt" "$R/SMOKE_VERDICTS.json"; exit 0
fi
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_delta_free --root "$R" --raw "${RAW%.md}.json" --seeds 1600-1607 --recipe batch \
  --floors "$REPO/docs/audits/2026-10-20/delta_free_floors_out.txt" --json-out "$REPO/DELTA_FREE_VERDICTS.json" \
  > "$REPO/DELTA_FREE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$MARK" "$REPO/DELTA_FREE_ANALYSIS.txt" "$REPO/DELTA_FREE_VERDICTS.json" "${RAW%.md}.json"
