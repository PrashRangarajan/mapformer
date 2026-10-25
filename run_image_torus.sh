#!/usr/bin/env bash
# IMAGE_TORUS_PREREG.md: the headline torus (64x64, T = 1024, 900 epochs, FACTORIAL's recipe) with each observation
# token replaced by an image of its class (MNIST digits + six Fashion-MNIST classes; train split in training, test
# split at evaluation). Seeds 3100-3107 (fresh). Block A (registered core): I_MapWM, I_ActOnly, I_RoPE, T_MapWM,
# T_ActOnly = 40 runs (T_ActOnly added by Amendment 1); Amendment 2 adds I_MapWM_g1 (image MapWM, encoder gain 1.0)
# = 48 runs, and P1's path arm becomes I_ActOnly;
# block B (optional, declared before launch): I_NormStep, I_RoPE_L4, T_RoPE = 24 runs.
# REQUIRED in the environment (no defaults): DRV_GPUS, DRV_MAXPG, IMG_BLOCKS (A or AB; must equal the prereg's
# "Declared blocks:" line). Planned launch (after both registered batches finish, after commit and the checker's amendment):
#   cd /home/prashr/mapformer && IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4 setsid nohup bash run_image_torus.sh > /dev/null 2>&1 &
# Dry run (own run dir and log; skips the clean-tree / committed / marker checks with a logged warning):
#   cd /home/prashr/mapformer && DRV_DRYRUN=1 IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4 bash run_image_torus.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
DRY="${DRV_DRYRUN:-0}"
if [ "$DRY" = 1 ]; then LOG="$REPO/image_torus_dryrun.log"; R="$REPO/runs/image_torus_dryrun"
else LOG="$REPO/image_torus.log"; R="$REPO/runs/image_torus"; fi
REQ_GPUS="${DRV_GPUS:-}"; REQ_MAXPG="${DRV_MAXPG:-}"; BLK="${IMG_BLOCKS:-}"   # captured BEFORE lib_driver.sh's defaults
source "$REPO/lib_driver.sh"
[ -n "$REQ_GPUS" ] && [ -n "$REQ_MAXPG" ] && [ -n "$BLK" ] ||
  { _drv_log "REFUSED: set DRV_GPUS, DRV_MAXPG and IMG_BLOCKS explicitly"; exit 1; }
[[ "$REQ_GPUS" =~ ^[0-9]+( [0-9]+)*$ && "$REQ_MAXPG" =~ ^[1-8]$ && "$BLK" =~ ^(A|AB)$ ]] ||
  { _drv_log "REFUSED: bad DRV_GPUS '$REQ_GPUS' / DRV_MAXPG '$REQ_MAXPG' / IMG_BLOCKS '$BLK'"; exit 1; }
DECL=$(sed -n 's/^Declared blocks: \(A\|AB\)$/\1/p' "$REPO/IMAGE_TORUS_PREREG.md" | tail -1)   # the LAST one (Amendment 1, m6)
[ "$DECL" = "$BLK" ] || { _drv_log "REFUSED: IMG_BLOCKS=$BLK but IMAGE_TORUS_PREREG.md declares '$DECL'"; exit 1; }
DRV_GPUS="$REQ_GPUS"; export DRV_GPUS
DRV_MAXPG="$REQ_MAXPG"; DRV_MINFREE=4000; DRV_SPACING=30     # assigned after sourcing (rule 27)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_image_torus.lock" || exit 1
echo "start $(date)" >> "$LOG"
for mk in .ssm_sign_a2_done .factorial_n8_done; do              # the two registered batches of 2026-10-25 must be done
  if [ ! -f "$REPO/$mk" ]; then
    if [ "$DRY" = 1 ] || [ "${DRV_SKIP_MARKERS:-0}" = 1 ]; then _drv_log "WARNING: $mk missing (dry run / DRV_SKIP_MARKERS=1)"
    else _drv_log "REFUSED: $mk missing (a registered batch may still be running)"; exit 1; fi
  fi
done
if (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: tracked files modified (a real launch would refuse)" ||
    { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
fi
cd "$REPO/.."
SEEDS=(3100 3101 3102 3103 3104 3105 3106 3107)
# (block AB puts the deep cell first within a seed: it is the longest)
# cell = name:arm:obs:layers:encoder gain ("-" = default 30.3729; the gain is part of the cell name when not default)
CELLS_A=("I_ActOnly:ActOnly:image:1:-" "I_MapWM:MapWM:image:1:-" "I_RoPE:RoPE:image:1:-" "I_MapWM_g1:MapWM:image:1:1.0"
         "T_MapWM:MapWM:token:1:-" "T_ActOnly:ActOnly:token:1:-")    # T_ActOnly: Amendment 1, m3; I_MapWM_g1: Amendment 2
CELLS_B=("I_RoPE_L4:RoPE:image:4:-" "I_NormStep:NormStep:image:1:-" "T_RoPE:RoPE:token:1:-")
if [ "$BLK" = AB ]; then CELLS=("${CELLS_B[0]}" "${CELLS_A[@]}" "${CELLS_B[1]}" "${CELLS_B[2]}"); else CELLS=("${CELLS_A[@]}"); fi
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_image, mapformer.eval_image_torus, mapformer.analyze_image_torus, mapformer.gate_image_torus
import mapformer.data_parallel, mapformer.image_bank, mapformer.model_image
import mapformer.model_codes   # ActOnly / NormStep classes, imported lazily by model_image (Amendment 1, M1)
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
for f in model_codes.py model_image.py train_variant.py model_rank.py; do
  printf '%s\n' "${GUARD[@]}" | grep -qx "$f" || { _drv_log "ABORT: $f missing from the md5 guard"; exit 1; }
done
GUARD+=(lib_driver.sh run_image_torus.sh IMAGE_TORUS_PREREG.md IMAGE_TORUS_GATE.json IMAGE_TORUS_GATE.md test_image_torus.py)
DATA=(data/image_torus/bank16.npz data/image_torus/raw/mnist/train-images-idx3-ubyte.gz
      data/image_torus/raw/mnist/t10k-images-idx3-ubyte.gz data/image_torus/raw/fashion/train-images-idx3-ubyte.gz
      data/image_torus/raw/fashion/t10k-images-idx3-ubyte.gz)                   # gitignored: md5-guarded, not committed
if ! (cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: a guarded file is not committed (a real launch would refuse)" ||
    { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
fi
drv_md5_guard "$R" "${GUARD[@]}" "${DATA[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); blocks $BLK; GPUs $DRV_GPUS; MAXPG $DRV_MAXPG; md5 guard over $((${#GUARD[@]} + ${#DATA[@]})) files"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
COMMON=(--epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --schedule cosine --data-workers 3 --fast-attn)
REQ=()
for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do         # seed outer, cell inner: a partial batch stays paired
  IFS=: read -r NAME ARM OBS NL GAIN <<< "$C"; OUT="$R/$NAME/${NAME}_s$S"; REQ+=("$OUT/$NAME.pt")
  GA=(); [ "$GAIN" = - ] || GA=(--enc-gain "$GAIN")
  [ -f "$OUT/$NAME.pt" ] && { _drv_log "skip $NAME s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $NAME s$S (running)"; continue; }
  drv_md5_guard "$R" "${GUARD[@]}" "${DATA[@]}" || drv_fail "code changed during the batch"   # per launch (rule 22)
  if [ "$DRY" = 1 ]; then G="${DRV_GPUS%% *}"; else G=$(drv_wait_slot); fi   # a dry run never waits for a slot
  mkdir -p "$OUT"; _drv_log "$NAME s$S -> cuda:$G"
  OMP_NUM_THREADS=3 drv_launch "$R/$NAME/${NAME}_s${S}.log" python3 -u -m mapformer.train_image --arm "$ARM" --obs "$OBS" \
    --n-layers "$NL" "${GA[@]}" --seed "$S" "${COMMON[@]}" --device "cuda:$G" --output-dir "$OUT"
done; done
NEXP=$(( ${#SEEDS[@]} * ${#CELLS[@]} ))
[ "${#REQ[@]}" -eq "$NEXP" ] || drv_fail "expected $NEXP checkpoints, listed ${#REQ[@]}"
if [ "$DRY" = 1 ]; then
  _drv_log "DRYRUN eval: python3 -u -m mapformer.eval_image_torus --skip-done --device cuda:G --ckpt <${NEXP} checkpoints split over: $DRV_GPUS>"
  _drv_log "DRYRUN analysis: CUDA_VISIBLE_DEVICES= python3 -u -m mapformer.analyze_image_torus --blocks $BLK > IMAGE_TORUS_ANALYSIS.txt"
  _drv_log "DRYRUN complete"; exit 0
fi
sleep 60
DRV_WAIT_POLL=120 drv_wait_dir "$R/"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" "${DATA[@]}" || drv_fail "code changed before eval"
# evaluation: the checkpoint list split round-robin over DRV_GPUS, one detached evaluator per GPU
read -r -a GL <<< "$DRV_GPUS"
for gi in "${!GL[@]}"; do
  PART=(); for i in "${!REQ[@]}"; do [ $(( i % ${#GL[@]} )) -eq "$gi" ] && PART+=("${REQ[$i]}"); done
  [ "${#PART[@]}" -gt 0 ] || continue
  OMP_NUM_THREADS=3 setsid nohup python3 -u -m mapformer.eval_image_torus --skip-done --device "cuda:${GL[$gi]}" \
    --ckpt "${PART[@]}" > "$R/eval_gpu${GL[$gi]}.log" 2>&1 &
  _drv_log "eval: ${#PART[@]} runs on cuda:${GL[$gi]} (pid $!)"
done
sleep 30
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
EVJ=(); for f in "${REQ[@]}"; do EVJ+=("$(dirname "$f")/eval_image.json"); done
drv_require "${EVJ[@]}" || drv_fail "missing eval JSONs"
drv_md5_guard "$R" "${GUARD[@]}" "${DATA[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_image_torus --blocks "$BLK" > "$REPO/IMAGE_TORUS_ANALYSIS.txt" 2>&1 ||
  drv_fail "analysis (exit status nonzero: VOID or error; read IMAGE_TORUS_ANALYSIS.txt)"
drv_done "$REPO/.image_torus_done" "$REPO/IMAGE_TORUS_ANALYSIS.txt" "$REPO/IMAGE_TORUS_VERDICTS.json" "${EVJ[@]}"
