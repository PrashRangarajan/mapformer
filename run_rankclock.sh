#!/usr/bin/env bash
# RANKCLOCK_PREREG.md batch (G-1, the rank clock-route test): torus T = 1024, 900 epochs, seeds 1900-1907 x arms
# Vanilla (shared r2), Vanilla_r2clockxy (Amendment 1: per-axis slot, the primary), Vanilla_r2clock, Vanilla_r2DF,
# Vanilla_r2AO, Vanilla_r4mi, and Vanilla_r4mi at weight decay 0 (own root, so the evaluators' <variant>_s<seed> layout holds).
# 56 runs, geometry snapshots every 10 epochs. RANKCLOCK_PREREG.md is md5-guarded: it must be FINAL before launch.
# GPU 1 only (the user's instruction, 2026-10-21); two jobs at a time (measured: 2 concurrent 4.2 s/epoch, 3 concurrent
# 6.1 s/epoch -- a third adds ~3% throughput at 100% utilisation). ~29 h.
#   cd /home/prashr/mapformer && setsid nohup bash run_rankclock.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rankclock.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=2; DRV_MINFREE=4500; DRV_SPACING=20           # assigned after sourcing (rule 27)
DRV_GPUS=1; export DRV_GPUS                              # unconditional (Amendment 1 finding 8): GPU 1 only
export CUDA_DEVICE_ORDER=PCI_BUS_ID                      # torch's cuda:1 = nvidia-smi's GPU 1
drv_lock "$REPO/.run_rankclock.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch model_rankclock.py analyze_rankclock.py run_rankclock.sh test_rankclock.py \
   construct_rankclock.py run_rankclock_existence.sh test_epoch_callback.py RANKCLOCK_PREREG.md train.py train_variant.py \
   > /dev/null 2>&1) ||
  { _drv_log "REFUSED: a RANKCLOCK file is not committed"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/rankclock"; RW="$REPO/runs/rankclock_wd0"; mkdir -p "$R/p0" "$RW/p0"
SEEDS=(1900 1901 1902 1903 1904 1905 1906 1907)
ARMS=(Vanilla Vanilla_r2clockxy Vanilla_r2clock Vanilla_r2DF Vanilla_r2AO Vanilla_r4mi WD0)
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.analyze_rankclock
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_rankclock.sh RANKCLOCK_PREREG.md)   # ckpt_guard: imported lazily by the evaluator
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
COMMON=(--epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128
        --n-landmarks 0 --schedule cosine --data-workers 3 --save-full-state --snapshot-every 10)
# seed-major: a partial batch is balanced across arms
for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do
  if [ "$A" = WD0 ]; then V=Vanilla_r4mi; OUT="$RW/p0/${V}_s$S"; WDA=(--weight-decay 0); TAG="r4wd0"
  else V=$A; OUT="$R/p0/${V}_s$S"; WDA=(--weight-decay 0.05); TAG="$A"; fi
  [ -f "$OUT/${V}.pt" ] && { _drv_log "skip $TAG s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $TAG s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$TAG s$S -> cuda:$G"
  OMP_NUM_THREADS=4 drv_launch "$OUT.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    "${COMMON[@]}" "${WDA[@]}" --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"; DRV_WAIT_POLL=60 drv_wait_dir "$RW/"
REQ=(); for S in "${SEEDS[@]}"; do
  for V in Vanilla Vanilla_r2clockxy Vanilla_r2clock Vanilla_r2DF Vanilla_r2AO Vanilla_r4mi; do REQ+=("$R/p0/${V}_s$S/${V}.pt"); done
  REQ+=("$RW/p0/Vanilla_r4mi_s$S/Vanilla_r4mi.pt"); done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 56 ] || drv_fail "expected 56 checkpoints, found $NF"
NS=$(ls "$R"/p0/*/snaps/e0900.pt "$RW"/p0/*/snaps/e0900.pt 2>/dev/null | wc -l)
[ "$NS" -eq 56 ] || drv_fail "expected 56 final snapshots, found $NS"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla Vanilla_r2clockxy Vanilla_r2clock Vanilla_r2DF \
  Vanilla_r2AO Vanilla_r4mi --noises 0.0 --seeds "${SEEDS[@]}" --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "RANKCLOCK (G-1): rank clock routes, trained and tested at T=1024" --out "$REPO/RANKCLOCK.md" >> "$LOG" 2>&1 \
  || drv_fail eval
python3 -u -m mapformer.eval_noise_refine --runs-dir "$RW" --variants Vanilla_r4mi --noises 0.0 --seeds "${SEEDS[@]}" \
  --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "RANKCLOCK (G-1): shared r4 at weight decay 0, T=1024" --out "$REPO/RANKCLOCK_WD0.md" >> "$LOG" 2>&1 \
  || drv_fail eval_wd0
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_rankclock > "$REPO/RANKCLOCK_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rankclock_done" "$REPO/RANKCLOCK.md" "$REPO/RANKCLOCK.json" "$REPO/RANKCLOCK_WD0.md" \
  "$REPO/RANKCLOCK_WD0.json" "$REPO/RANKCLOCK_ANALYSIS.txt" "$REPO/RANKCLOCK_VERDICTS.json"
