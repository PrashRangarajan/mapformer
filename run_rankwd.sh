#!/usr/bin/env bash
# RANKWD_PREREG.md batch: is weight decay necessary for shared rank 4 on the T = 1024 torus? Seeds 2100-2107 x
# Vanilla_r4mi at weight decay 0.05 and at 0 (own root each, so the evaluators' <variant>_s<seed> layout holds).
# 16 runs, the RANK_MI / RANKCLOCK recipe, geometry snapshots every 10 epochs. GPU 1 only, two jobs at a time (~9 h).
# RANKWD_PREREG.md is md5-guarded: it must be FINAL before launch.
#   cd /home/prashr/mapformer && setsid nohup bash run_rankwd.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rankwd.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=2; DRV_MINFREE=4500; DRV_SPACING=20           # assigned after sourcing (rule 27)
DRV_GPUS=1; export DRV_GPUS                              # GPU 1 only (the user's instruction)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rankwd.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch analyze_rankwd.py analyze_rankclock.py model_rankclock.py run_rankwd.sh \
   RANKWD_PREREG.md train.py train_variant.py > /dev/null 2>&1) || { _drv_log "REFUSED: a RANKWD file is not committed"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/rankwd"; RW="$REPO/runs/rankwd_wd0"; mkdir -p "$R/p0" "$RW/p0"
SEEDS=(2100 2101 2102 2103 2104 2105 2106 2107)
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.analyze_rankwd
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_rankwd.sh RANKWD_PREREG.md)   # ckpt_guard: imported lazily by the evaluator
# every guarded file must be COMMITTED, not merely unmodified (Amendment 1 finding 7: the clean-tree check skips untracked)
(cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1) || { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
COMMON=(--variant Vanilla_r4mi --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2
        --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --save-full-state --snapshot-every 10)
# seed-major, arms alternating: a partial batch stays paired
for S in "${SEEDS[@]}"; do for A in WD WD0; do
  if [ "$A" = WD0 ]; then OUT="$RW/p0/Vanilla_r4mi_s$S"; WDV=0; else OUT="$R/p0/Vanilla_r4mi_s$S"; WDV=0.05; fi
  [ -f "$OUT/Vanilla_r4mi.pt" ] && { _drv_log "skip $A s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A s$S -> cuda:$G"
  OMP_NUM_THREADS=4 drv_launch "$OUT.log" python3 -u -m mapformer.train_variant "${COMMON[@]}" --seed "$S" \
    --weight-decay "$WDV" --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"; DRV_WAIT_POLL=60 drv_wait_dir "$RW/"
REQ=(); for S in "${SEEDS[@]}"; do REQ+=("$R/p0/Vanilla_r4mi_s$S/Vanilla_r4mi.pt" "$RW/p0/Vanilla_r4mi_s$S/Vanilla_r4mi.pt"); done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 16 ] || drv_fail "expected 16 checkpoints, found $NF"
NS=$(ls "$R"/p0/*/snaps/e0900.pt "$RW"/p0/*/snaps/e0900.pt 2>/dev/null | wc -l)
[ "$NS" -eq 16 ] || drv_fail "expected 16 final snapshots, found $NS"
NA=$(ls "$R"/p0/*/snaps/e*.pt "$RW"/p0/*/snaps/e*.pt 2>/dev/null | wc -l)     # X2b reads every snapshot (finding 7)
[ "$NA" -eq 1456 ] || drv_fail "expected 16 x 91 = 1456 snapshots, found $NA"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
python3 -u -m mapformer.eval_noise_refine --runs-dir "$R" --variants Vanilla_r4mi --noises 0.0 --seeds "${SEEDS[@]}" \
  --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "RANKWD: shared r4 at weight decay 0.05, T=1024" --out "$REPO/RANKWD.md" >> "$LOG" 2>&1 || drv_fail eval
python3 -u -m mapformer.eval_noise_refine --runs-dir "$RW" --variants Vanilla_r4mi --noises 0.0 --seeds "${SEEDS[@]}" \
  --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "RANKWD: shared r4 at weight decay 0, T=1024" --out "$REPO/RANKWD_WD0.md" >> "$LOG" 2>&1 || drv_fail eval_wd0
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_rankwd > "$REPO/RANKWD_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rankwd_done" "$REPO/RANKWD.md" "$REPO/RANKWD.json" "$REPO/RANKWD_WD0.md" "$REPO/RANKWD_WD0.json" \
  "$REPO/RANKWD_ANALYSIS.txt" "$REPO/RANKWD_VERDICTS.json"
