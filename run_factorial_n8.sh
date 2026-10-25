#!/usr/bin/env bash
# FACTORIAL_N8_PREREG.md: fresh-seed replication of FACTORIAL -- RoPE at 1 layer / 4 layers / looped x4 and MapWM (Vanilla_r4)
# at 1 layer on the torus, T = 1024, 900 epochs, --fast-attn, FACTORIAL's exact flags; seeds 2800-2803 (fresh). 16 runs.
# DRV_GPUS and DRV_MAXPG are REQUIRED in the environment (no defaults). Planned launch (kalman, GPU 1, 4 jobs at a time):
#   cd /home/prashr/mapformer && DRV_GPUS=1 DRV_MAXPG=4 setsid nohup bash run_factorial_n8.sh > /dev/null 2>&1 &
# Dry run (logs launches only; own run dir and log, skips the clean-tree / committed checks with a logged warning):
#   cd /home/prashr/mapformer && DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=4 bash run_factorial_n8.sh
# FACTORIAL_N8_PREREG.md is md5-guarded: it must be FINAL before launch.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
DRY="${DRV_DRYRUN:-0}"
if [ "$DRY" = 1 ]; then LOG="$REPO/factorial_n8_dryrun.log"; R="$REPO/runs/factorial_n8_dryrun"
else LOG="$REPO/factorial_n8.log"; R="$REPO/runs/factorial_n8"; fi
REQ_GPUS="${DRV_GPUS:-}"; REQ_MAXPG="${DRV_MAXPG:-}"          # captured BEFORE lib_driver.sh assigns its defaults
source "$REPO/lib_driver.sh"
[ -n "$REQ_GPUS" ] && [ -n "$REQ_MAXPG" ] || { _drv_log "REFUSED: set DRV_GPUS and DRV_MAXPG explicitly (e.g. DRV_GPUS=1 DRV_MAXPG=4)"; exit 1; }
[[ "$REQ_GPUS" =~ ^[0-9]+( [0-9]+)*$ && "$REQ_MAXPG" =~ ^[1-8]$ ]] || { _drv_log "REFUSED: bad DRV_GPUS '$REQ_GPUS' / DRV_MAXPG '$REQ_MAXPG'"; exit 1; }
DRV_GPUS="$REQ_GPUS"; export DRV_GPUS
DRV_MAXPG="$REQ_MAXPG"; DRV_MINFREE=4000; DRV_SPACING=30     # assigned after sourcing (rule 27); FACTORIAL's values
export CUDA_DEVICE_ORDER=PCI_BUS_ID                           # cuda:N == nvidia-smi index N (drv_freemem reads nvidia-smi)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_factorial_n8.lock" || exit 1
echo "start $(date)" >> "$LOG"
if (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: tracked files modified (a real launch would refuse)" ||
    { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
fi
cd "$REPO/.."
SEEDS=(2800 2801 2802 2803)
# cell = root:variant:layers -- FACTORIAL's four cells; deep cells first within a seed (as FACTORIAL)
CELLS=("L4:RoPE:4" "LOOP:RoPELooped:1" "L1:RoPE:1" "L1:Vanilla_r4:1")
for c in L1 L4 LOOP; do mkdir -p "$R/$c/p0"; done
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.analyze_factorial, mapformer.analyze_factorial_n8
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_factorial_n8.sh FACTORIAL_N8_PREREG.md
        FACTORIAL_L1.json FACTORIAL_L4.json FACTORIAL_LOOP.json                       # FACTORIAL's evals: the pooled secondary
        FACTORIAL_L1_STRATA.json FACTORIAL_L4_STRATA.json FACTORIAL_LOOP_STRATA.json)  # reads them (and its checkpoints)
if ! (cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: a guarded file is not committed (a real launch would refuse)" ||
    { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
fi
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS; MAXPG $DRV_MAXPG; md5 guard over ${#GUARD[@]} files"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
# FACTORIAL's exact training flags (run_factorial.sh); only seed / device / output dir differ
COMMON=(--epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-heads 2 --d-model 128 --n-landmarks 0
        --schedule cosine --data-workers 3 --fast-attn)
for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do      # seed outer, cell inner: a partial batch stays paired by seed
  IFS=: read -r ROOT V NL <<< "$C"; OUT="$R/$ROOT/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $ROOT $V s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $ROOT $V s$S (running)"; continue; }
  drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed during the batch"      # per launch (rule 22)
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$ROOT $V L$NL s$S -> cuda:$G"
  OMP_NUM_THREADS=3 drv_launch "$R/$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    "${COMMON[@]}" --n-layers "$NL" --device "cuda:$G" --output-dir "$OUT"
done; done
EV(){ if [ "$DRY" = 1 ]; then _drv_log "DRYRUN eval: $*"; else "$@" >> "$LOG" 2>&1; fi; }
if [ "$DRY" != 1 ]; then
  sleep 60
  DRV_WAIT_POLL=120 drv_wait_dir "$R/"
  REQ=(); for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do IFS=: read -r ROOT V NL <<< "$C"; REQ+=("$R/$ROOT/p0/${V}_s$S/$V.pt"); done; done
  [ "${#REQ[@]}" -eq 16 ] || drv_fail "expected 16 checkpoints"
  drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
  drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
fi
G=$(drv_wait_slot)
for spec in "L1:RoPE Vanilla_r4" "L4:RoPE" "LOOP:RoPELooped"; do ROOT=${spec%%:*}; VS=${spec#*:}
  EV python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 512 1024 2048 --n-trials 100 --device "cuda:$G" --title "Factorial n8 (fresh) $ROOT" \
    --out "$REPO/FACTORIAL_N8_${ROOT}.md" || drv_fail "eval $ROOT"
  EV python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/$ROOT" --variants $VS --seeds "${SEEDS[@]}" --lengths 1024 \
    --n-trials 100 --device "cuda:$G" --check-json "$REPO/FACTORIAL_N8_${ROOT}.json" \
    --out "$REPO/FACTORIAL_N8_${ROOT}_STRATA.json" || drv_fail "strata $ROOT"
  for S in "${SEEDS[@]}"; do        # FACTORIAL Amendment 1 (D3): each run on its OWN training map (env seed = run seed), secondary
    EV python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds "$S" --env-seed "$S" \
      --lengths 1024 --n-trials 100 --device "cuda:$G" --title "Factorial n8 $ROOT own map s$S" \
      --out "$REPO/FACTORIAL_N8_${ROOT}_OWN_s${S}.md" || _drv_log "own-map eval $ROOT s$S failed (secondary)"
  done
done
[ "$DRY" = 1 ] && { _drv_log "DRYRUN analysis: CUDA_VISIBLE_DEVICES= python3 -u -m mapformer.analyze_factorial_n8 > FACTORIAL_N8_ANALYSIS.txt"; _drv_log "DRYRUN complete"; exit 0; }
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_factorial_n8 > "$REPO/FACTORIAL_N8_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.factorial_n8_done" "$REPO/FACTORIAL_N8_ANALYSIS.txt" "$REPO/FACTORIAL_N8_VERDICTS.json" \
  "$REPO/FACTORIAL_N8_L1.json" "$REPO/FACTORIAL_N8_L4.json" "$REPO/FACTORIAL_N8_LOOP.json"
