#!/usr/bin/env bash
# G3_SSM_PREREG.md batch (G-3, the sign test in a one-layer diagonal complex SSM). NOT LAUNCHED by the preparing session.
#   lr 3e-3 (fixed from the signed pilots, G3_SSM_PREREG.md Sec. 5) except block A2.
#   Stage 1  block A: seeds 2100-2107 x m {32, 8} x arms signed abs softplus exp index, steps = moves    (80 runs, 300 ep)
#            block A2: the same five arms at lr 1e-3, m = 32 only                                      (40 runs, 300 ep)
#                     (G3_BLOCK_A2=0 drops it; the analysis is then run with --no-a2)
#            block B: seeds 2100-2107 x arms signed abs softplus exp, steps = all, m = 32             (32 runs, 300 ep)
#                     (G3_BLOCK_B=0 drops it; the analysis is then run with --no-b)
#   Stage 2  block H: per seed, construct_ssm_sign.py from block A's signed m = 32 run (CPU), then 100-epoch continuations
#                     of the abs / softplus / exp constructions and the signed control                  (32 runs, 100 ep)
#            block H2: the SAME constructions continued at lr 1e-3 (with A2; G3_BLOCK_A2=0 drops both)   (32 runs, 100 ep)
# Seed-major order, so a partial batch is balanced across arms. One shared 4090 by default (DRV_GPUS=1), at most 4 of our
# runs per GPU (DRV_MAXPG=4; measured beside the other user's jobs: 4 concurrent runs 5.0 s/epoch each vs 2.1 s alone).
#   cd /home/prashr/mapformer && setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/ssm_sign.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${G3_MAXPG:-4}"; DRV_MINFREE=2500; DRV_SPACING=5      # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-1}"
BLOCK_B="${G3_BLOCK_B:-1}"
BLOCK_A2="${G3_BLOCK_A2:-1}"
drv_lock "$REPO/.run_ssm_sign.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/ssm_sign/p0"; mkdir -p "$R/A" "$R/A2" "$R/B" "$R/H/built" "$R/H2"
SEEDS=(2100 2101 2102 2103 2104 2105 2106 2107)
SIZES=(32 8)
ARMS_A=(signed abs softplus exp index)
ARMS_B=(signed abs softplus exp)
CLASSES_H=(abs softplus exp signed)
FILES=(model_ssm_sign.py train_ssm_sign.py construct_ssm_sign.py analyze_ssm_sign.py gate_ssm_sign.py test_ssm_sign.py
       run_ssm_sign.sh G3_SSM_PREREG.md)
echo "start $(date)" >> "$LOG"

clean_tree() {   # tracked files unmodified AND every batch file committed (rule 7: the script behind every number)
  (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) &&
    { _drv_log "REFUSED: tracked files modified (commit first)"; return 1; }
  (cd "$REPO" && git ls-files --error-unmatch "${FILES[@]}" > /dev/null 2>&1) ||
    { _drv_log "REFUSED: a G-3 file is not committed"; return 1; }
}
clean_tree || exit 1
# md5 guard over EVERY mapformer module the batch imports (as run_signroute.sh); data_parallel is imported lazily by train()
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_ssm_sign, mapformer.construct_ssm_sign, mapformer.data_parallel, mapformer.analyze_ssm_sign
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 15 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_ssm_sign.sh G3_SSM_PREREG.md)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS max/GPU $DRV_MAXPG; block A2 $BLOCK_A2 block B $BLOCK_B; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
launch() {   # $1 out dir, $2 lr, rest = trainer args
  local OUT="$1" LR="$2"; shift 2
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $OUT (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $OUT (running)"; return 0; }
  local G; G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$OUT -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ssm_sign "$@" --lr "$LR" --device "cuda:$G" \
    --output-dir "$OUT"
}

# ---- stage 1 (seed-major; A before B within the batch so stage 2's sources finish first)
for S in "${SEEDS[@]}"; do for M in "${SIZES[@]}"; do for A in "${ARMS_A[@]}"; do
  launch "$R/A/${A}_moves_m${M}_s$S" 3e-3 --arm "$A" --steps moves --size "$M" --seed "$S"
done; done; done
if [ "$BLOCK_A2" = 1 ]; then
  for S in "${SEEDS[@]}"; do for A in "${ARMS_A[@]}"; do
    launch "$R/A2/${A}_moves_m32_lr1e-3_s$S" 1e-3 --arm "$A" --steps moves --size 32 --seed "$S"
  done; done
fi
if [ "$BLOCK_B" = 1 ]; then
  for S in "${SEEDS[@]}"; do for A in "${ARMS_B[@]}"; do
    launch "$R/B/${A}_all_m32_s$S" 3e-3 --arm "$A" --steps all --size 32 --seed "$S"
  done; done
fi

# ---- stage 2: constructions need every signed m = 32 source
clean_tree || drv_fail "tree changed during stage 1"
sleep 30
DRV_WAIT_POLL=60 drv_wait_dir "$R/A/signed_moves_m32_"
SRC=(); for S in "${SEEDS[@]}"; do SRC+=("$R/A/signed_moves_m32_s$S/signed.pt" "$R/A/signed_moves_m32_s$S/eval.json"); done
drv_require "${SRC[@]}" || drv_fail "missing signed m = 32 sources"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the constructions"
for S in "${SEEDS[@]}"; do
  B="$R/H/built/s$S"
  if [ ! -f "$B/report.json" ]; then
    mkdir -p "$B"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 -u -m mapformer.construct_ssm_sign \
      --source "$R/A/signed_moves_m32_s$S/signed.pt" --save "$B" --json "$B/report.json" > "$B/construct.log" 2>&1 ||
      drv_fail "construction s$S"
  fi
  for C in "${CLASSES_H[@]}"; do drv_require "$B/${C}_built.pt" || drv_fail "missing $B/${C}_built.pt"; done
  for C in "${CLASSES_H[@]}"; do
    launch "$R/H/${C}_built_moves_m32_s$S" 3e-3 --arm "$C" --steps moves --size 32 --seed "$S" --epochs 100 \
      --init-built "$B/${C}_built.pt"
    [ "$BLOCK_A2" = 1 ] && launch "$R/H2/${C}_built_moves_m32_lr1e-3_s$S" 1e-3 --arm "$C" --steps moves --size 32 \
      --seed "$S" --epochs 100 --init-built "$B/${C}_built.pt"
  done
done

# ---- wait, verify every artifact, analyse
clean_tree || drv_fail "tree changed during stage 2"
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=()
for S in "${SEEDS[@]}"; do
  for M in "${SIZES[@]}"; do for A in "${ARMS_A[@]}"; do
    REQ+=("$R/A/${A}_moves_m${M}_s$S/eval.json" "$R/A/${A}_moves_m${M}_s$S/${A}.pt"); done; done
  if [ "$BLOCK_A2" = 1 ]; then for A in "${ARMS_A[@]}"; do
    REQ+=("$R/A2/${A}_moves_m32_lr1e-3_s$S/eval.json" "$R/A2/${A}_moves_m32_lr1e-3_s$S/${A}.pt"); done; fi
  if [ "$BLOCK_B" = 1 ]; then for A in "${ARMS_B[@]}"; do
    REQ+=("$R/B/${A}_all_m32_s$S/eval.json" "$R/B/${A}_all_m32_s$S/${A}.pt"); done; fi
  for C in "${CLASSES_H[@]}"; do
    REQ+=("$R/H/${C}_built_moves_m32_s$S/eval.json" "$R/H/${C}_built_moves_m32_s$S/${C}.pt" "$R/H/built/s$S/${C}_built.pt")
    [ "$BLOCK_A2" = 1 ] && REQ+=("$R/H2/${C}_built_moves_m32_lr1e-3_s$S/eval.json" "$R/H2/${C}_built_moves_m32_lr1e-3_s$S/${C}.pt")
  done
done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NWANT=$(( ${#SEEDS[@]} * (2 * ${#SIZES[@]} * ${#ARMS_A[@]} + 3 * ${#CLASSES_H[@]}) + BLOCK_B * ${#SEEDS[@]} * 2 * ${#ARMS_B[@]} + BLOCK_A2 * ${#SEEDS[@]} * 2 * (${#ARMS_A[@]} + ${#CLASSES_H[@]}) ))
[ "${#REQ[@]}" -eq "$NWANT" ] || drv_fail "expected $NWANT artifacts, listed ${#REQ[@]}"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
NOB=(); [ "$BLOCK_B" = 1 ] || NOB+=(--no-b); [ "$BLOCK_A2" = 1 ] || NOB+=(--no-a2)
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_ssm_sign "${NOB[@]}" > "$REPO/G3_SSM_ANALYSIS.txt" 2>&1 ||
  drv_fail analyze
drv_done "$REPO/.ssm_sign_done" "$REPO/G3_SSM_ANALYSIS.txt" "$REPO/G3_SSM_VERDICTS.json"
