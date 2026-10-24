#!/usr/bin/env bash
# RANKRECIPE_PREREG.md batch: does shared rank 2 still fail the T = 1024 torus at its own best recipe? Seeds 2600-2607 x
# cells r2_R0 / r2_R1 / r2_R2 (shared rank 2 at three recipes) + r4_R0 (shared rank 4 at the default), the RANKWD recipe
# otherwise, 900 epochs, snapshots every 10 epochs, own root per cell. 32 runs. GPU 1 only, two jobs at a time (~17 h).
# Recipes are read from analyze_rankrecipe.RECIPES so driver and analysis cannot disagree (R1's lr is set by the pilot rule).
# RANKRECIPE_PREREG.md is md5-guarded: it must be FINAL before launch.
#   cd /home/prashr/mapformer && setsid nohup bash run_rankrecipe.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rankrecipe.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=2; DRV_MINFREE=4500; DRV_SPACING=20           # assigned after sourcing (rule 27)
DRV_GPUS=1; export DRV_GPUS                              # GPU 1 only (the user's instruction)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rankrecipe.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/rankrecipe"
SEEDS=(2600 2601 2602 2603 2604 2605 2606 2607)
CELLS=(r2_R0 r4_R0 r2_R1 r2_R2)
declare -A VAR LR WD
while read -r C V L W; do VAR[$C]=$V; LR[$C]=$L; WD[$C]=$W; done < <(CUDA_VISIBLE_DEVICES="" python3 -c "
from mapformer.analyze_rankrecipe import CELLS, RECIPES
for c, (v, r) in CELLS.items():
    print(c, v, repr(RECIPES[r][0]), repr(RECIPES[r][1]))" 2>> "$LOG")
for C in "${CELLS[@]}"; do [ -n "${LR[$C]:-}" ] || { _drv_log "ABORT: no recipe for $C"; exit 1; }; mkdir -p "$R/$C/p0"; done
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.analyze_rankrecipe
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_rankrecipe.sh RANKRECIPE_PREREG.md)
(cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1) || { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files; recipes $(for C in "${CELLS[@]}"; do printf '%s=%s/lr%s/wd%s ' "$C" "${VAR[$C]}" "${LR[$C]}" "${WD[$C]}"; done)"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
COMMON=(--epochs 900 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0
        --schedule cosine --data-workers 3 --save-full-state --snapshot-every 10)
# seed-major, cells in turn: a partial batch stays paired by seed
for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do
  V="${VAR[$C]}"; OUT="$R/$C/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $C s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $C s$S (running)"; continue; }
  drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed during the batch"      # per launch (rule 22)
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$C s$S -> cuda:$G"
  OMP_NUM_THREADS=4 drv_launch "$OUT.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" "${COMMON[@]}" \
    --lr "${LR[$C]}" --weight-decay "${WD[$C]}" --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
for C in "${CELLS[@]}"; do DRV_WAIT_POLL=60 drv_wait_dir "$R/$C/"; done
REQ=(); for C in "${CELLS[@]}"; do for S in "${SEEDS[@]}"; do REQ+=("$R/$C/p0/${VAR[$C]}_s$S/${VAR[$C]}.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 32 ] || drv_fail "expected 32 checkpoints, found $NF"
for E in e0000 e0900; do N=$(ls "$R"/*/p0/*/snaps/$E.pt 2>/dev/null | wc -l); [ "$N" -eq 32 ] || drv_fail "expected 32 $E snapshots, found $N"; done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for C in "${CELLS[@]}"; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$C" --variants "${VAR[$C]}" --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
    --title "RANKRECIPE: $C, T=1024" --out "$REPO/RANKRECIPE_$C.md" >> "$LOG" 2>&1 || drv_fail "eval $C"
done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_rankrecipe > "$REPO/RANKRECIPE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rankrecipe_done" "$REPO/RANKRECIPE_ANALYSIS.txt" "$REPO/RANKRECIPE_VERDICTS.json" \
  "$REPO/RANKRECIPE_r2_R0.json" "$REPO/RANKRECIPE_r2_R1.json" "$REPO/RANKRECIPE_r2_R2.json" "$REPO/RANKRECIPE_r4_R0.json"
