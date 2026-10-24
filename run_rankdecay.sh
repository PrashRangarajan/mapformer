#!/usr/bin/env bash
# RANKDECAY_PREREG.md batch: which parameters must AdamW decay for shared rank 4 on the T = 1024 torus? Seeds 2600-2607 x
# arms all / none / noomega / omega (decay on omega x decay on the rest; wd 0.05 where on), Vanilla_r4mi, the RANKWD recipe,
# 32 runs, geometry snapshots every 10 epochs, own root per arm. GPU 1 only, two jobs at a time (~17 h).
# RANKDECAY_PREREG.md is md5-guarded: it must be FINAL before launch.
#   cd /home/prashr/mapformer && setsid nohup bash run_rankdecay.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rankdecay.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=2; DRV_MINFREE=4500; DRV_SPACING=20           # assigned after sourcing (rule 27)
DRV_GPUS=1; export DRV_GPUS                              # GPU 1 only (the user's instruction)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
drv_lock "$REPO/.run_rankdecay.lock" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date)" >> "$LOG"
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/rankdecay"
SEEDS=(2600 2601 2602 2603 2604 2605 2606 2607)
ARMS=(all none noomega omega)
declare -A WDA=([all]="--weight-decay 0.05" [none]="--weight-decay 0"
                [noomega]="--weight-decay 0.05 --wd-select exclude --wd-names path_integrator.omega"
                [omega]="--weight-decay 0.05 --wd-select only --wd-names path_integrator.omega")
for A in "${ARMS[@]}"; do mkdir -p "$R/$A/p0"; done
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.data_parallel, mapformer.eval_noise_refine, mapformer.analyze_rankdecay
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_rankdecay.sh RANKDECAY_PREREG.md test_wd_groups.py)
(cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1) || { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
COMMON=(--variant Vanilla_r4mi --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2
        --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --save-full-state --snapshot-every 10)
# seed-major, arms in turn: a partial batch stays balanced across arms
for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do
  OUT="$R/$A/p0/Vanilla_r4mi_s$S"
  [ -f "$OUT/Vanilla_r4mi.pt" ] && { _drv_log "skip $A s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A s$S -> cuda:$G"
  # shellcheck disable=SC2086  -- WDA values are flag lists by design
  OMP_NUM_THREADS=4 drv_launch "$OUT.log" python3 -u -m mapformer.train_variant "${COMMON[@]}" --seed "$S" ${WDA[$A]} \
    --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
for A in "${ARMS[@]}"; do DRV_WAIT_POLL=60 drv_wait_dir "$R/$A/"; done
REQ=(); for A in "${ARMS[@]}"; do for S in "${SEEDS[@]}"; do REQ+=("$R/$A/p0/Vanilla_r4mi_s$S/Vanilla_r4mi.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 32 ] || drv_fail "expected 32 checkpoints, found $NF"
NS=$(ls "$R"/*/p0/*/snaps/e0900.pt 2>/dev/null | wc -l)
[ "$NS" -eq 32 ] || drv_fail "expected 32 final snapshots, found $NS"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for A in "${ARMS[@]}"; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$A" --variants Vanilla_r4mi --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
    --title "RANKDECAY: shared r4, arm $A, T=1024" --out "$REPO/RANKDECAY_$A.md" >> "$LOG" 2>&1 || drv_fail "eval $A"
done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_rankdecay > "$REPO/RANKDECAY_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rankdecay_done" "$REPO/RANKDECAY_ANALYSIS.txt" "$REPO/RANKDECAY_VERDICTS.json" \
  "$REPO/RANKDECAY_all.json" "$REPO/RANKDECAY_none.json" "$REPO/RANKDECAY_noomega.json" "$REPO/RANKDECAY_omega.json"
