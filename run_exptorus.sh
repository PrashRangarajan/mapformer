#!/usr/bin/env bash
# EXPTORUS_PREREG.md: the exp and softplus phase increments in ATTENTION on the T = 1024 torus, beside the signed and |.|
# anchors, each arm at its own best of lr 1e-3 / 3e-3 (the G-4 TORUS_LRSWEEP machinery). Recipe = G-4 / GENERALITY E3's
# torus recipe unchanged (train_variant via train_exptorus: observation-step-free rank 4, batch 16, 98 batches, T 1024,
# 900 epochs, cosine, --fast-attn, --data-workers 3, OMP 2, AdamW wd 0.05, 1 layer, 2 heads, d 128). Arms SO SignedAO4_r4,
# AO AbsAO4_r4, EXP ExpAO4_r4, SPL SoftplusAO4_r4; fresh seeds 3400-3407: 4 arms x 2 rates x 8 seeds = 64 runs. Then evals
# (eval_noise_refine T 1024 / 2048, eval_rank_strata T 1024, both through eval_exptorus) and the registered analysis (CPU).
# kalman only. DRV_GPUS and DRV_MAXPG MUST be given at launch (no defaults):
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_exptorus.sh > /dev/null 2>&1 &
# Modes (MODE=..., one lock, so never two at once):
#   batch  (default) the 64-run batch, evals, analysis, marker .exptorus_done. Refuses unless the tree is clean for tracked
#          *.py / *.sh and the EXPTORUS files are committed.
#   pilot  exp at 1e-3 and 3e-3, seeds 3490-3491 (outside the batch, never pooled), 900 epochs (4 runs), then
#          `analyze_exptorus --pilot` (timing, UNSTABLE, census); marker .exptorus_pilot_done. May run before the commit
#          (ALLOW_UNCOMMITTED=1): the md5 record and `git status` are then kept in the pilot dir.
#   smoke  CPU only, tiny sizes (3 epochs, 2 batches of 2, T 64), seeds 3498-3499, all 4 arms x 2 rates, evals and the
#          analysis with --smoke; marker .exptorus_smoke_done. Exercises the batch's train / eval / analysis command lines
#          on CPU (not the GPU launch path, not `analyze_exptorus --pilot`).
# Amendment 1 (docs/audits/2026-10-26/exptorus_audit.md):
#   - batch and pilot REFUSE to start while any mapformer.train_ process outside this mode's run dir runs, while
#     .run_rnn_torus.lock or .run_rnn_torus_pilot.lock is held, or while another user has a GPU compute process; every
#     launch's occupancy is appended to <run dir>/launch_occupancy.txt (the pilot analysis reads it for the 12 h rule)
#   - the batch's rates come from the ONE line `EXPTORUS_RATES = ...` in EXPTORUS_PREREG.md (committed), so a switch to
#     option A is an amendment line, not a code change; the pilot always runs both rates (it is the input to that rule)
#   - the pilot is launched by hand (the earlier waiter script is retired)
# The md5 guard is re-checked before EVERY launch: no git pull / checkout / edit of a guarded module (train_variant.py, the
# model modules it imports, the evaluators, the analysis, lib_driver.sh, this file) while a mode is running.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-batch}"
case "$MODE" in batch) LOG="$REPO/exptorus.log" ;; pilot) LOG="$REPO/exptorus_pilot.log" ;; smoke) LOG="$REPO/exptorus_smoke.log" ;;
  *) echo "$(date '+%F %T') ABORT: MODE=$MODE" >> "$REPO/exptorus.log"; exit 1 ;; esac
if [ "$MODE" != smoke ] && { [ -z "${DRV_GPUS:-}" ] || [ -z "${DRV_MAXPG:-}" ]; }; then
  echo "$(date '+%F %T') REFUSED: set DRV_GPUS and DRV_MAXPG at launch (e.g. DRV_GPUS=\"0 1\" DRV_MAXPG=2)" >> "$LOG"; exit 1
fi
_GPUS="${DRV_GPUS:-0}"; _MAXPG="${DRV_MAXPG:-1}"
source "$REPO/lib_driver.sh"
DRV_GPUS="$_GPUS"; DRV_MAXPG="$_MAXPG"; DRV_MINFREE=3000; DRV_SPACING=20   # assigned after sourcing (rule 27)
export DRV_GPUS
[[ "$DRV_MAXPG" =~ ^[1-2]$ ]] || { _drv_log "REFUSED: DRV_MAXPG=$DRV_MAXPG (1-2: the timing estimate is at 2 per GPU)"; exit 1; }
for g in $DRV_GPUS; do [[ "$g" =~ ^[0-9]$ ]] || { _drv_log "REFUSED: DRV_GPUS=$DRV_GPUS"; exit 1; }; done
[ "$(hostname)" = kalman ] || [ "$MODE" = smoke ] || { _drv_log "REFUSED: EXPTORUS runs on kalman only (host $(hostname))"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID                  # torch's cuda:N = nvidia-smi's GPU N (drv_pick reads nvidia-smi)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_exptorus.lock" || exit 1
echo "start $(date) MODE=$MODE GPUs $DRV_GPUS maxpg $DRV_MAXPG host $(hostname)" >> "$LOG"
OWN=(model_exptorus.py train_exptorus.py eval_exptorus.py analyze_exptorus.py test_exptorus.py run_exptorus.sh EXPTORUS_PREREG.md)
DEPS=(model_signroute.py model_wrap_exist.py model_sign.py train_variant.py eval_noise_refine.py eval_rank_strata.py
      analyze_torus_lrsweep.py analyze_lrsweep.py analyze_robust.py stats_core.py lib_driver.sh)
DIRTY=$(cd "$REPO" && git status --porcelain --untracked-files=no -- '*.py' '*.sh' EXPTORUS_PREREG.md)
COMMITTED=1; (cd "$REPO" && git ls-files --error-unmatch "${OWN[@]}" "${DEPS[@]}" > /dev/null 2>&1) || COMMITTED=0
if [ "$MODE" = batch ]; then
  [ -n "$DIRTY" ] && { _drv_log "REFUSED: modified tracked code / prereg (commit first): $(echo $DIRTY)"; exit 1; }
  [ "$COMMITTED" = 1 ] || { _drv_log "REFUSED: an EXPTORUS file or a dependency is not committed"; exit 1; }
elif [ "$COMMITTED" = 0 ] || [ -n "$DIRTY" ]; then
  [ "${ALLOW_UNCOMMITTED:-0}" = 1 ] || { _drv_log "REFUSED: $MODE on an uncommitted / dirty tree needs ALLOW_UNCOMMITTED=1"; exit 1; }
  _drv_log "WARNING: $MODE runs on an uncommitted tree (ALLOW_UNCOMMITTED=1); git status kept in the run dir"
fi
cd "$REPO/.."
R="$REPO/runs/exptorus"; RP="$REPO/runs/exptorus_pilot"; RS="$REPO/runs/exptorus_smoke"
case "$MODE" in batch) RT="$R"; SEEDS=(3400 3401 3402 3403 3404 3405 3406 3407) ;;
  pilot) RT="$RP"; SEEDS=(3490 3491) ;; smoke) RT="$RS"; SEEDS=(3498 3499) ;; esac
EV="$RT/eval"
ARMS=(SO AO EXP SPL)
if [ "$MODE" = batch ]; then   # Amendment 1.5: declared in the prereg, exactly one line, 1e-3 always included
  NDECL=$(grep -c '^EXPTORUS_RATES = ' "$REPO/EXPTORUS_PREREG.md")
  [ "$NDECL" = 1 ] || { _drv_log "REFUSED: EXPTORUS_PREREG.md must declare exactly one 'EXPTORUS_RATES = ...' line ($NDECL)"; exit 1; }
  read -r -a RATES <<< "$(grep '^EXPTORUS_RATES = ' "$REPO/EXPTORUS_PREREG.md" | sed 's/^EXPTORUS_RATES = //')"
  case "${RATES[*]}" in "1e-3 3e-3"|"1e-3") ;; *) _drv_log "REFUSED: declared rates '${RATES[*]}' (allowed: '1e-3 3e-3' or '1e-3')"; exit 1 ;; esac
else
  RATES=(1e-3 3e-3)
fi
_drv_log "rates: ${RATES[*]}"
declare -A TAG=([1e-3]=lr1e-03 [3e-3]=lr3e-03)    # = analyze_torus_lrsweep.rate_tag
declare -A VAR=([SO]=SignedAO4_r4 [AO]=AbsAO4_r4 [EXP]=ExpAO4_r4 [SPL]=SoftplusAO4_r4)
for L in "${!TAG[@]}"; do
  [ "${TAG[$L]}" = "$(python3 -c "print(f'lr{float(\"$L\"):.0e}')")" ] || { _drv_log "ABORT: tag mismatch for $L"; exit 1; }
done
# md5 guard over every mapformer module the trainer, evaluators and analysis import (sys.modules; data_parallel and
# ckpt_guard are imported lazily, so explicitly), plus the driver and lib_driver
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_exptorus, mapformer.eval_exptorus, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.data_parallel, mapformer.ckpt_guard, mapformer.analyze_exptorus
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
for f in model_exptorus.py train_exptorus.py eval_exptorus.py analyze_exptorus.py train_variant.py model_signroute.py; do
  printf '%s\n' "${GUARD[@]}" | grep -qx "$f" || { _drv_log "ABORT: $f missing from the md5 guard list"; exit 1; }
done
GUARD+=(lib_driver.sh run_exptorus.sh)
drv_md5_guard "$RT" "${GUARD[@]}" || exit 1
(cd "$REPO" && { git rev-parse HEAD; git status --porcelain --untracked-files=no; }) > "$RT/git_state_$(date +%Y%m%d_%H%M%S).txt"
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) (committed=$COMMITTED); md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
occupancy(){  # one line: trainers per GPU, trainers outside $RT, other users' GPU compute processes (Amendment 1.3)
  local g per="" own foreign others=0 p u
  for g in $DRV_GPUS; do per+=" g$g=$(drv_ntrain "$g")"; done
  foreign=$(ps -u "$USER" -o comm=,args= | awk -v r="--output-dir $RT/" '$1=="python3" && /mapformer\.train_/ && !index($0, r)' | wc -l)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" != "$USER" ] && others=$((others + 1))
  done
  echo "$(date '+%F %T')${per} foreign_trainers=$foreign other_users=$others"
}
if [ "$MODE" != smoke ]; then   # Amendment 1.3: sequential with RNN_TORUS and any other job; refuse, never share
  for LK in "$REPO/.run_rnn_torus.lock" "$REPO/.run_rnn_torus_pilot.lock"; do
    if [ -e "$LK" ] && ! flock -n "$LK" true; then _drv_log "REFUSED: $LK is held (RNN_TORUS running)"; exit 1; fi
  done
  OCC=$(occupancy)
  case "$OCC" in *"foreign_trainers=0 other_users=0") ;; *) _drv_log "REFUSED: GPUs not free: $OCC"; exit 1 ;; esac
  _drv_log "occupancy at start: $OCC"
fi

if [ "$MODE" = smoke ]; then   # CPU, sequential, tiny; the batch's command lines with sizes shrunk
  EP=3; NB=2
  for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
    V="${VAR[$A]}"; OUT="$RT/${TAG[$L]}/$A/p0/${V}_s$S"
    [ -f "$OUT/$V.pt" ] && continue
    mkdir -p "$OUT"; _drv_log "smoke $L $A s$S"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python3 -u -m mapformer.train_exptorus --variant "$V" --seed "$S" --lr "$L" \
      --weight-decay 0.05 --n-batches $NB --batch-size 2 --n-steps 64 --n-layers 1 --n-heads 2 --d-model 128 \
      --n-landmarks 0 --schedule cosine --data-workers 0 --fast-attn --epochs $EP --device cpu --output-dir "$OUT" \
      > "$RT/${TAG[$L]}/${A}_s$S.log" 2>&1 || drv_fail "smoke train $L $A s$S"
  done; done; done
  mkdir -p "$EV"
  for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
    CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.eval_exptorus noise_refine --runs-dir "$RT/${TAG[$L]}/$A" \
      --variants "${VAR[$A]}" --noises 0.0 --seeds "${SEEDS[@]}" --lengths 64 128 --n-trials 4 --device cpu \
      --title "EXPTORUS smoke lr $L $A" --out "$EV/EXPTORUS_${TAG[$L]}_$A.md" >> "$LOG" 2>&1 || drv_fail "smoke eval $L $A"
    CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.eval_exptorus strata --runs-dir "$RT/${TAG[$L]}/$A" --variants "${VAR[$A]}" \
      --seeds "${SEEDS[@]}" --lengths 64 --n-trials 4 --device cpu --check-json "$EV/EXPTORUS_${TAG[$L]}_$A.json" \
      --out "$EV/EXPTORUS_${TAG[$L]}_${A}_STRATA.json" >> "$LOG" 2>&1 || _drv_log "WARNING: smoke strata $L $A failed"
  done; done
  drv_md5_guard "$RT" "${GUARD[@]}" || drv_fail "code changed during the smoke"
  CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_exptorus --smoke --root "$RT" --seeds 3498-3499 --epochs $EP \
    --n-batches $NB --lengths 64,128 --json-out "$RT/SMOKE_VERDICTS.json" > "$RT/SMOKE_ANALYSIS.txt" 2>&1 || drv_fail "smoke analysis"
  drv_done "$REPO/.exptorus_smoke_done" "$RT/SMOKE_ANALYSIS.txt" "$RT/SMOKE_VERDICTS.json"
  exit 0
fi

torus(){  # root rate arm seed
  local RT_="$1" L="$2" A="$3" S="$4" V="${VAR[$3]}" OUT G
  OUT="$RT_/${TAG[$L]}/$A/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $L $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $L $A s$S (running)"; return 0; }
  G=$(drv_wait_slot "$DRV_MAXPG")
  drv_md5_guard "$RT_" "${GUARD[@]}" || drv_fail "code changed during the batch (before launching $L $A s$S)"
  mkdir -p "$OUT"; _drv_log "torus $L $A s$S -> cuda:$G"
  echo "launch $L $A s$S cuda:$G $(occupancy)" >> "$RT_/launch_occupancy.txt"
  OMP_NUM_THREADS=2 drv_launch "$RT_/${TAG[$L]}/${A}_s$S.log" python3 -u -m mapformer.train_exptorus --variant "$V" --seed "$S" \
    --lr "$L" --weight-decay 0.05 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 \
    --n-landmarks 0 --schedule cosine --data-workers 3 --fast-attn --epochs 900 --device "cuda:$G" --output-dir "$OUT"
}

if [ "$MODE" = pilot ]; then
  for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do torus "$RT" "$L" EXP "$S"; done; done
  sleep 60; DRV_WAIT_POLL=60 drv_wait_dir "$RT/"
  REQ=(); for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do REQ+=("$RT/${TAG[$L]}/EXP/p0/ExpAO4_r4_s$S/ExpAO4_r4.pt"); done; done
  drv_require "${REQ[@]}" || drv_fail "missing pilot runs"
  drv_md5_guard "$RT" "${GUARD[@]}" || drv_fail "code changed before the pilot analysis"
  CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_exptorus --pilot --pilot-root "$RT" --pilot-seeds 3490-3491 \
    --batch-runs 64 --slots 4 > "$RT/PILOT_ANALYSIS.txt" 2>&1 || drv_fail "pilot analysis"
  drv_done "$REPO/.exptorus_pilot_done" "$RT/PILOT_ANALYSIS.txt"
  exit 0
fi

# batch: seed outer, rate, arm inner (a partial batch is balanced across rates and arms)
for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do torus "$RT" "$L" "$A" "$S"; done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$RT/"
REQ=(); for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  REQ+=("$RT/${TAG[$L]}/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NEXP=$(( ${#ARMS[@]} * ${#RATES[@]} * ${#SEEDS[@]} ))
[ "${#REQ[@]}" -eq "$NEXP" ] || drv_fail "expected $NEXP checkpoints, listed ${#REQ[@]}"
drv_md5_guard "$RT" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
mkdir -p "$EV"; EVDEV="cuda:${DRV_GPUS%% *}"
for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do
  python3 -u -m mapformer.eval_exptorus noise_refine --runs-dir "$RT/${TAG[$L]}/$A" --variants "${VAR[$A]}" --noises 0.0 \
    --seeds "${SEEDS[@]}" --lengths 1024 2048 --n-trials 100 --device "$EVDEV" --title "EXPTORUS lr $L $A" \
    --out "$EV/EXPTORUS_${TAG[$L]}_$A.md" >> "$LOG" 2>&1 || drv_fail "eval $L $A"
  # declared secondary: a strata failure is logged only (the analysis reports it unavailable)
  python3 -u -m mapformer.eval_exptorus strata --runs-dir "$RT/${TAG[$L]}/$A" --variants "${VAR[$A]}" --seeds "${SEEDS[@]}" \
    --lengths 1024 --n-trials 100 --device "$EVDEV" --check-json "$EV/EXPTORUS_${TAG[$L]}_$A.json" \
    --out "$EV/EXPTORUS_${TAG[$L]}_${A}_STRATA.json" >> "$LOG" 2>&1 || _drv_log "WARNING: strata $L $A failed (declared secondary)"
done; done
EVREQ=(); for L in "${RATES[@]}"; do for A in "${ARMS[@]}"; do EVREQ+=("$EV/EXPTORUS_${TAG[$L]}_$A.json"); done; done
drv_require "${EVREQ[@]}" || drv_fail "missing evals"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_exptorus --root "$RT" --seeds 3400-3407 \
  --rates "$(IFS=,; echo "${RATES[*]}")" --json-out "$REPO/EXPTORUS_VERDICTS.json" > "$REPO/EXPTORUS_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.exptorus_done" "$REPO/EXPTORUS_ANALYSIS.txt" "$REPO/EXPTORUS_VERDICTS.json" "${EVREQ[@]}"
