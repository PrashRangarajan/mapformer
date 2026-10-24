#!/usr/bin/env bash
# TORUS_LRSWEEP_PREREG.md (G-4): the sign learnability cost on the T = 1024 torus with each step parameterisation at its own
# best learning rate. GENERALITY's torus recipe (train_variant, observation-step-free SignedAO4_r4 (SO) vs AbsAO4_r4 (AO),
# batch 16, 98 batches, T 1024, 900 epochs, cosine, --fast-attn, --data-workers 3, OMP 2, AdamW wd 0.05) at lr 1e-3 / 3e-3 /
# 1e-2, fresh seeds 2400-2407: 2 arms x 3 rates x 8 seeds = 48 runs. Then evals (eval_noise_refine T 1024 / 2048,
# eval_rank_strata T 1024) and the registered analysis (CPU). No trainer flag is added (lr and --weight-decay exist).
# DRV_GPUS and DRV_MAXPG MUST be given at launch (no defaults: koopman runs GPU 1 only at 2 jobs).
#   cd /home/prashr/mapformer && DRV_GPUS=1 DRV_MAXPG=2 setsid nohup bash run_torus_lrsweep.sh > /dev/null 2>&1 &
# Modes (MODE=..., same lock, so never two at once):
#   batch   (default) the 48-run batch, evals, analysis, marker .torus_lrsweep_done
#   pilot   grid-extension pilot (prereg Sec. 8 + Amendment 1): AO at 1e-2 and 3e-2, seeds 2492-2493, 900 epochs (4 runs), then
#           `analyze_torus_lrsweep --pilot`; marker .torus_lrsweep_pilot_done. Run BEFORE the batch; its outcome goes into
#           an amendment (which edits RATES below if it says EXTEND).
#   stage2  the conditional decay-matched stage 2 (prereg Sec. 3 + Amendment 1): only if TORUS_LRSWEEP_VERDICTS.json says
#           stage2_triggered; SO / AO at AO's selected rate with wd = 5e-5 / lr, seeds 2400-2407 (16 runs), evals, analysis;
#           marker .torus_lrsweep_stage2_done. Amendment 1 MINOR 4: the TRAINING modules must be unchanged against the
#           batch's md5 record; stage 2 then keeps its own record under runs/torus_lrsweep/stage2 (analysis / evaluator /
#           driver fixes from stage 2's own check are allowed).
# The md5 guard is re-checked before EVERY launch (Amendment 1 MINOR 3): no git pull / checkout on the GPU machine while a
# mode is running.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-batch}"
case "$MODE" in batch) LOG="$REPO/torus_lrsweep.log" ;; pilot) LOG="$REPO/torus_lrsweep_pilot.log" ;;
  stage2) LOG="$REPO/torus_lrsweep_stage2.log" ;; *) echo "$(date '+%F %T') ABORT: MODE=$MODE" >> "$REPO/torus_lrsweep.log"; exit 1 ;; esac
if [ -z "${DRV_GPUS:-}" ] || [ -z "${DRV_MAXPG:-}" ]; then
  echo "$(date '+%F %T') REFUSED: set DRV_GPUS and DRV_MAXPG at launch (e.g. DRV_GPUS=1 DRV_MAXPG=2)" >> "$LOG"; exit 1
fi
_GPUS="$DRV_GPUS"; _MAXPG="$DRV_MAXPG"
source "$REPO/lib_driver.sh"
DRV_GPUS="$_GPUS"; DRV_MAXPG="$_MAXPG"; DRV_MINFREE=3000; DRV_SPACING=20   # assigned after sourcing (rule 27)
export DRV_GPUS
[[ "$DRV_MAXPG" =~ ^[1-4]$ ]] || { _drv_log "REFUSED: DRV_MAXPG=$DRV_MAXPG (1-4)"; exit 1; }
for g in $DRV_GPUS; do [[ "$g" =~ ^[0-9]$ ]] || { _drv_log "REFUSED: DRV_GPUS=$DRV_GPUS"; exit 1; }; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID                  # torch's cuda:N = nvidia-smi's GPU N (drv_pick reads nvidia-smi)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_torus_lrsweep.lock" || exit 1
echo "start $(date) MODE=$MODE GPUs $DRV_GPUS maxpg $DRV_MAXPG" >> "$LOG"
# Clean tree BEFORE any wait, restricted to what runs: no tracked *.py / *.sh and not the prereg may be modified (results
# notes and other docs may be); the G-4 files must be committed.
DIRTY=$(cd "$REPO" && git status --porcelain --untracked-files=no -- '*.py' '*.sh' TORUS_LRSWEEP_PREREG.md)
[ -n "$DIRTY" ] && { _drv_log "REFUSED: modified tracked code / prereg (commit first): $(echo $DIRTY)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch analyze_torus_lrsweep.py run_torus_lrsweep.sh TORUS_LRSWEEP_PREREG.md \
   analyze_lrsweep.py train_variant.py docs/paper/expansion/scripts/sign/clock_table.py > /dev/null 2>&1) ||
  { _drv_log "REFUSED: a G-4 file is not committed"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/torus_lrsweep"; RP="$REPO/runs/torus_lrsweep_pilot"; EV="$R/eval"
SEEDS=(2400 2401 2402 2403 2404 2405 2406 2407)
RATES=(1e-3 3e-3 1e-2)                      # an amendment extending the grid (pilot rule) edits RATES only
declare -A TAG=([1e-3]=lr1e-03 [3e-3]=lr3e-03 [1e-2]=lr1e-02 [3e-2]=lr3e-02)   # = analyze_torus_lrsweep.rate_tag
declare -A VAR=([SO]=SignedAO4_r4 [AO]=AbsAO4_r4)
for L in "${!TAG[@]}"; do
  [ "${TAG[$L]}" = "$(python3 -c "print(f'lr{float(\"$L\"):.0e}')")" ] || { _drv_log "ABORT: tag mismatch for $L"; exit 1; }
done
# md5 guard over every mapformer module the trainers, evaluators and analysis import (sys.modules; data_parallel and
# ckpt_guard are imported lazily, so explicitly), the clock / map script the analysis loads from its file, and the driver
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.ckpt_guard, mapformer.analyze_torus_lrsweep
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(docs/paper/expansion/scripts/sign/clock_table.py lib_driver.sh run_torus_lrsweep.sh)
GROOT="$R"; [ "$MODE" = pilot ] && GROOT="$RP"; [ "$MODE" = stage2 ] && GROOT="$R/stage2"
if [ "$MODE" = stage2 ]; then   # Amendment 1 MINOR 4: training modules unchanged since the batch
  [ -f "$R/code_md5.txt" ] || { _drv_log "REFUSED: stage 2 without the batch's md5 record"; exit 1; }
  mapfile -t TGUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
  [ "${#TGUARD[@]}" -ge 20 ] || { _drv_log "ABORT: training module list has ${#TGUARD[@]} entries"; exit 1; }
  for f in "${TGUARD[@]}"; do
    want=$(awk -v f="$f" '$2==f {print $1}' "$R/code_md5.txt"); have=$(cd "$REPO" && md5sum "$f" | awk '{print $1}')
    [ -n "$want" ] && [ "$want" = "$have" ] || { _drv_log "REFUSED: training module $f changed since the batch (or absent from its record)"; exit 1; }
  done
  _drv_log "stage 2: ${#TGUARD[@]} training modules unchanged against the batch's md5 record"
fi
drv_md5_guard "$GROOT" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD); md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
torus(){  # root rate arm seed weight_decay
  local RT="$1" L="$2" A="$3" S="$4" WD="$5" V="${VAR[$3]}" OUT G
  OUT="$RT/${TAG[$L]}/$A/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $L $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $L $A s$S (running)"; return 0; }
  G=$(drv_wait_slot "$DRV_MAXPG")
  drv_md5_guard "$GROOT" "${GUARD[@]}" || drv_fail "code changed during the batch (before launching $L $A s$S)"   # MINOR 3
  mkdir -p "$OUT"; _drv_log "torus $L $A s$S wd $WD -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$RT/${TAG[$L]}/${A}_s$S.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --lr "$L" --weight-decay "$WD" --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 \
    --n-landmarks 0 --schedule cosine --data-workers 3 --fast-attn --epochs 900 --device "cuda:$G" --output-dir "$OUT"
}
evals(){  # root rate arm prefix [strata]
  local RT="$1" L="$2" A="$3" P="$4" V="${VAR[$3]}" EVDEV="cuda:${DRV_GPUS%% *}"
  mkdir -p "$EV"
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$RT/${TAG[$L]}/$A" --variants "$V" --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 1024 2048 --n-trials 100 --device "$EVDEV" --title "TORUS_LRSWEEP $P lr $L $A" \
    --out "$EV/${P}_${TAG[$L]}_$A.md" >> "$LOG" 2>&1 || drv_fail "eval $L $A"
  if [ "${5:-}" = strata ]; then   # declared secondary: a failure (e.g. a non-finite model trips --check-json) is logged only
    python3 -u -m mapformer.eval_rank_strata --runs-dir "$RT/${TAG[$L]}/$A" --variants "$V" --seeds "${SEEDS[@]}" --lengths 1024 \
      --n-trials 100 --device "$EVDEV" --check-json "$EV/${P}_${TAG[$L]}_$A.json" --out "$EV/${P}_${TAG[$L]}_${A}_STRATA.json" \
      >> "$LOG" 2>&1 || _drv_log "WARNING: strata $L $A failed (declared secondary; the analysis reports it unavailable)"
  fi
}

if [ "$MODE" = pilot ]; then
  for S in 2492 2493; do for L in 1e-2 3e-2; do torus "$RP" "$L" AO "$S" 0.05; done; done
  sleep 60; DRV_WAIT_POLL=60 drv_wait_dir "$RP/"
  REQ=(); for S in 2492 2493; do for L in 1e-2 3e-2; do REQ+=("$RP/${TAG[$L]}/AO/p0/AbsAO4_r4_s$S/AbsAO4_r4.pt"); done; done
  drv_require "${REQ[@]}" || drv_fail "missing pilot runs"
  drv_md5_guard "$RP" "${GUARD[@]}" || drv_fail "code changed before the pilot analysis"
  CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_torus_lrsweep --pilot --pilot-root "$RP" --pilot-seeds 2492-2493 \
    > "$RP/PILOT_ANALYSIS.txt" 2>&1 || drv_fail "pilot analysis"
  drv_done "$REPO/.torus_lrsweep_pilot_done" "$RP/PILOT_ANALYSIS.txt"
  exit 0
fi

if [ "$MODE" = stage2 ]; then
  read -r TRIG LR2 < <(python3 -c "
import json; V = json.load(open('$REPO/TORUS_LRSWEEP_VERDICTS.json'))
print(int(bool(V['stage2_triggered'])), f\"{V['B1']['AO_rate']:.0e}\".replace('e-0', 'e-'))" 2>> "$LOG")
  [ "${TRIG:-0}" = 1 ] || { _drv_log "REFUSED: stage 2 is not triggered by TORUS_LRSWEEP_VERDICTS.json"; exit 1; }
  [ -n "${TAG[$LR2]:-}" ] || { _drv_log "ABORT: stage-2 rate $LR2 has no tag"; exit 1; }
  WD2=$(python3 -c "print(repr(5e-5 / float('$LR2')))")
  _drv_log "stage 2: lr $LR2, wd $WD2"
  for S in "${SEEDS[@]}"; do for A in SO AO; do torus "$R/stage2" "$LR2" "$A" "$S" "$WD2"; done; done
  sleep 60; DRV_WAIT_POLL=60 drv_wait_dir "$R/stage2/"
  REQ=(); for S in "${SEEDS[@]}"; do for A in SO AO; do REQ+=("$R/stage2/${TAG[$LR2]}/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt"); done; done
  drv_require "${REQ[@]}" || drv_fail "missing stage-2 runs"
  drv_md5_guard "$GROOT" "${GUARD[@]}" || drv_fail "code changed before the stage-2 evaluation"
  for A in SO AO; do evals "$R/stage2" "$LR2" "$A" TORUS_LRSWEEP_S2; done
  CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_torus_lrsweep --stage2 --root "$R" --seeds 2400-2407 \
    --json-out "$REPO/TORUS_LRSWEEP_VERDICTS.json" \
    > "$REPO/TORUS_LRSWEEP_STAGE2.txt" 2>&1 || drv_fail "stage-2 analysis"
  drv_done "$REPO/.torus_lrsweep_stage2_done" "$REPO/TORUS_LRSWEEP_STAGE2.txt"
  exit 0
fi

# batch: seed outer, rate, arm inner (a partial batch is balanced across rates and arms)
for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in SO AO; do torus "$R" "$L" "$A" "$S" 0.05; done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do for L in "${RATES[@]}"; do for A in SO AO; do
  REQ+=("$R/${TAG[$L]}/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NEXP=$(( 2 * ${#RATES[@]} * ${#SEEDS[@]} ))
[ "${#REQ[@]}" -eq "$NEXP" ] || drv_fail "expected $NEXP checkpoints, listed ${#REQ[@]}"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for L in "${RATES[@]}"; do for A in SO AO; do evals "$R" "$L" "$A" TORUS_LRSWEEP strata; done; done
EVREQ=(); for L in "${RATES[@]}"; do for A in SO AO; do EVREQ+=("$EV/TORUS_LRSWEEP_${TAG[$L]}_$A.json"); done; done
drv_require "${EVREQ[@]}" || drv_fail "missing evals"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_torus_lrsweep --root "$R" --seeds 2400-2407 \
  --rates "$(IFS=,; echo "${RATES[*]}")" --json-out "$REPO/TORUS_LRSWEEP_VERDICTS.json" > "$REPO/TORUS_LRSWEEP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.torus_lrsweep_done" "$REPO/TORUS_LRSWEEP_ANALYSIS.txt" "$REPO/TORUS_LRSWEEP_VERDICTS.json" "${EVREQ[@]}"
