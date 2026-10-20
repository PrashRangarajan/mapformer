#!/usr/bin/env bash
# SROPE_TORUS_PREREG.md batch, seeds 1700-1707: does faithful Selective RoPE learn the 2D torus map? Arms SRoPE (authors' best
# LM config), SRoPE_NoSiLU, SRoPE_ST (their state-tracking config, Amendment 1), SRoPEGen (model_selective, as is), MapWM = Signed_r4 (in-batch positive reference), RoPE (index,
# in-batch negative reference). GENERALITY's torus recipe: train_variant, T = 1024, 900 epochs, lr 1e-3, batch 16, 98 batches,
# cosine, --fast-attn, --data-workers 3, OMP 2, 1 layer, 2 heads, d 128. Then evals (eval_noise_refine T 1024 / 2048,
# eval_rank_strata T 1024) and the registered analysis. Concurrency: at most 4 of our trainers per GPU (the policy's cap on a
# GPU shared with another user = GENERALITY's normal concurrency for this recipe). GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_srope_torus.sh > /dev/null 2>&1 &
#   MODE=smoke SMOKE_DIR=/path bash run_srope_torus.sh   # CPU, 2 seeds (1780-1781), 2 epochs x 2 batches, 3 eval trials
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-batch}"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=6000; DRV_SPACING=20    # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(SRoPE SRoPE_NoSiLU SRoPE_ST SRoPEGen MapWM RoPE)   # SRoPE_ST: Amendment 1
declare -A VAR=([SRoPE]=SRoPE [SRoPE_NoSiLU]=SRoPE_NoSiLU [SRoPE_ST]=SRoPE_ST [SRoPEGen]=SRoPEGen [MapWM]=Signed_r4 [RoPE]=RoPE)
if [ "$MODE" = batch ]; then
  LOG="$REPO/srope_torus.log"; R="$REPO/runs/srope_torus"; EV="$REPO"; SEEDS=(1700 1701 1702 1703 1704 1705 1706 1707)
  EPOCHS=900; NB=98; NTRIALS=100; WALKS=8; ABL=100; LOCK="$REPO/.run_srope_torus.lock"
elif [ "$MODE" = smoke ]; then
  : "${SMOKE_DIR:?set SMOKE_DIR}"
  LOG="$SMOKE_DIR/smoke.log"; R="$SMOKE_DIR/runs"; EV="$SMOKE_DIR"; SEEDS=(1780 1781)
  EPOCHS=2; NB=2; NTRIALS=3; WALKS=1; ABL=3; LOCK="$SMOKE_DIR/.lock"; DRV_SPACING=0; mkdir -p "$SMOKE_DIR"
else
  echo "MODE must be batch or smoke" >&2; exit 2
fi
drv_lock "$LOCK" || exit 1
echo "start $MODE $(date)" >> "$LOG"
# clean tree BEFORE any wait (a batch must run committed code); smoke runs on the working tree and says so
if [ "$MODE" = batch ]; then
  (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
  (cd "$REPO" && git ls-files --error-unmatch analyze_srope_torus.py run_srope_torus.sh model_srope.py SROPE_TORUS_PREREG.md \
      > /dev/null 2>&1) || { _drv_log "REFUSED: analyze_srope_torus.py / run_srope_torus.sh / model_srope.py / prereg not committed"; exit 1; }
else
  _drv_log "smoke: working tree $(cd "$REPO" && git rev-parse --short HEAD)$(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q . && echo ' + uncommitted edits')"
fi
cd "$REPO/.."
mkdir -p "$R"
# md5 guard over EVERY mapformer module the batch imports (sys.modules after importing trainer, evaluators, analysis;
# data_parallel is imported lazily inside train(), so explicitly), plus the driver files
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.data_parallel, mapformer.ckpt_guard, mapformer.analyze_srope_torus
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_srope_torus.sh)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD); mode $MODE; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
EVDEV="cuda:${DRV_GPUS%% *}"; [ "$MODE" = smoke ] && EVDEV=cpu
train(){  # arm seed
  local A="$1" S="$2" V="${VAR[$1]}" OUT="$R/$1/p0/${VAR[$1]}_s$2" G DEV
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; return 0; }
  if [ "$MODE" = smoke ]; then DEV=cpu; else G=$(drv_wait_slot); DEV="cuda:$G"; fi
  mkdir -p "$OUT"; _drv_log "$A s$S -> $DEV"
  OMP_NUM_THREADS=2 drv_launch "$R/${A}_s$S.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --lr 1e-3 --n-batches "$NB" --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --data-workers 3 --fast-attn --epochs "$EPOCHS" --device "$DEV" --output-dir "$OUT"
}
# seed outer, arm inner
for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do train "$A" "$S"; done; done
[ "$MODE" = batch ] && sleep 60
DRV_WAIT_POLL=$([ "$MODE" = batch ] && echo 60 || echo 5) drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do REQ+=("$R/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq $(( ${#SEEDS[@]} * ${#ARMS[@]} )) ] || drv_fail "expected $(( ${#SEEDS[@]} * ${#ARMS[@]} )) checkpoints, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for A in "${ARMS[@]}"; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$A" --variants "${VAR[$A]}" --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 1024 2048 --n-trials "$NTRIALS" --device "$EVDEV" --title "SROPE_TORUS $A" --out "$EV/SROPE_TORUS_$A.md" \
    >> "$LOG" 2>&1 || drv_fail "eval $A"
  python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/$A" --variants "${VAR[$A]}" --seeds "${SEEDS[@]}" --lengths 1024 \
    --n-trials "$NTRIALS" --device "$EVDEV" --check-json "$EV/SROPE_TORUS_$A.json" --out "$EV/SROPE_TORUS_${A}_STRATA.json" \
    >> "$LOG" 2>&1 || drv_fail "strata $A"
done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_srope_torus --root "$R" --eval-dir "$EV" \
  --seeds "${SEEDS[0]}-${SEEDS[-1]}" --epochs "$EPOCHS" --n-batches "$NB" --stats-walks "$WALKS" --ablate-trials "$ABL" \
  --json-out "$EV/SROPE_TORUS_VERDICTS.json" > "$EV/SROPE_TORUS_ANALYSIS.txt" 2>&1 || drv_fail analyze
MK="$REPO/.srope_torus_done"; [ "$MODE" = smoke ] && MK="$SMOKE_DIR/.srope_torus_done"
drv_done "$MK" "$EV/SROPE_TORUS_ANALYSIS.txt" "$EV/SROPE_TORUS_VERDICTS.json"
