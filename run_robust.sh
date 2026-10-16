#!/usr/bin/env bash
# ROBUST_PREREG.md batch, seeds 1400-1407. R2 torus FIRST (longest): SignedAO4_r4 / AbsAO4_r4 at d 256, 2 layers, 2 heads,
# T = 1024, 900 epochs, GENERALITY's torus recipe (train_variant, lr 1e-3, batch 16, 98 batches, cosine, --fast-attn,
# --data-workers 3, OMP 2), ONE run per GPU (measured: GPU-bound, 3.6 s/epoch alone, 16 s/epoch at 4 per GPU). Then R1 ring
# (train_gauge, +1/-1, RING_DIAL recipe) SO / AO in four recipes def / wd0 / lrx3 / lrd3, up to 6 trainers per GPU. Then the
# torus evals and the registered analysis. GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_robust.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/robust.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27)
TORUS_MAXPG=1; RING_MAXPG=6
export DRV_GPUS="${DRV_GPUS:-0 1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_robust.lock" || exit 1
echo "start $(date)" >> "$LOG"
# clean tree BEFORE any wait (a batch must run committed code)
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
(cd "$REPO" && git ls-files --error-unmatch analyze_robust.py run_robust.sh train_gauge.py > /dev/null 2>&1) ||
  { _drv_log "REFUSED: analyze_robust.py / run_robust.sh / train_gauge.py not committed"; exit 1; }
cd "$REPO/.."
R="$REPO/runs/robust"; mkdir -p "$R/ring" "$R/torus"
SEEDS=(1400 1401 1402 1403 1404 1405 1406 1407)
# md5 guard over EVERY mapformer module the batch imports (sys.modules after importing the trainers, evaluators and the
# analysis; data_parallel is imported lazily inside train(), so it is imported explicitly), plus the driver files
mapfile -t GUARD < <(CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_gauge, mapformer.train_variant, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.data_parallel, mapformer.ckpt_guard, mapformer.analyze_robust
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_robust.sh)
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS; md5 guard over ${#GUARD[@]} files"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
declare -A VAR=([SO]=SignedAO4_r4 [AO]=AbsAO4_r4)
declare -A LR=([def]=1e-3 [wd0]=1e-3 [lrx3]=3e-3 [lrd3]=3.333333e-4)
declare -A WD=([def]=0.05 [wd0]=0 [lrx3]=0.05 [lrd3]=0.05)
torus(){  # arm seed
  local A="$1" S="$2" V="${VAR[$1]}" OUT="$R/torus/$1/p0/${VAR[$1]}_s$2" G
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip torus $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip torus $A s$S (running)"; return 0; }
  G=$(drv_wait_slot "$TORUS_MAXPG"); mkdir -p "$OUT"; _drv_log "torus $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$R/torus/${A}_s$S.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 2 --n-heads 2 --d-model 256 --n-landmarks 0 \
    --schedule cosine --data-workers 3 --fast-attn --epochs 900 --device "cuda:$G" --output-dir "$OUT"
}
ring(){  # recipe arm seed
  local C="$1" A="$2" S="$3" OUT="$R/ring/$1/$2_s$3" G
  [ -f "$OUT/eval.json" ] && { _drv_log "skip ring $C $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip ring $C $A s$S (running)"; return 0; }
  G=$(drv_wait_slot "$RING_MAXPG"); mkdir -p "$OUT"; _drv_log "ring $C $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_gauge --arm "$A" --deltas=1,-1 --seed "$S" \
    --lr "${LR[$C]}" --weight-decay "${WD[$C]}" --device "cuda:$G" --output-dir "$OUT"
}
# R2 first (seed outer, arm inner), then R1 (seed outer, recipe, arm inner)
for S in "${SEEDS[@]}"; do for A in SO AO; do torus "$A" "$S"; done; done
for S in "${SEEDS[@]}"; do for C in def wd0 lrx3 lrd3; do for A in SO AO; do ring "$C" "$A" "$S"; done; done; done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do
  for A in SO AO; do REQ+=("$R/torus/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt")
    for C in def wd0 lrx3 lrd3; do REQ+=("$R/ring/$C/${A}_s$S/eval.json" "$R/ring/$C/${A}_s$S/$A.pt"); done
  done
done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 144 ] || drv_fail "expected 144 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for A in SO AO; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/torus/$A" --variants "${VAR[$A]}" --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 1024 2048 --n-trials 100 --device cuda:0 --title "ROBUST torus $A (d 256, 2 layers)" \
    --out "$REPO/ROBUST_TORUS_$A.md" >> "$LOG" 2>&1 || drv_fail "eval $A"
  python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/torus/$A" --variants "${VAR[$A]}" --seeds "${SEEDS[@]}" --lengths 1024 \
    --n-trials 100 --device cuda:0 --check-json "$REPO/ROBUST_TORUS_$A.json" --out "$REPO/ROBUST_TORUS_${A}_STRATA.json" \
    >> "$LOG" 2>&1 || drv_fail "strata $A"
done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_robust > "$REPO/ROBUST_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.robust_done" "$REPO/ROBUST_ANALYSIS.txt" "$REPO/ROBUST_VERDICTS.json"
