#!/usr/bin/env bash
# GENERALITY_PREREG.md batch: seeds 700-707. Torus: SO4, AO4, Signed_r4, CARoPEF from scratch (900 ep); per seed, once
# SO4 is done: build the constructions, then SO_bal / AO_bal (lr 1e-4) and SO_bal_hi / AO_bal_hi (lr 1e-3), 300 ep. Ring:
# Signed_r4, CARoPEF, CoPEF (300 ep). Then evals and the registered analysis. GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_generality.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/generality.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=6000; DRV_SPACING=20    # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_generality.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/generality"; mkdir -p "$R/torus" "$R/ring"
SEEDS=(700 701 702 703 704 705 706 707)
GUARD=(model_wrap_exist.py model_mono_baselines.py build_torus_exist.py train_variant.py train_ring_variant.py model.py
       model_sign.py lie_groups.py train.py environment.py environment_cancel.py environment_nd.py data_parallel.py
       train_cancel.py eval_noise_refine.py eval_rank_strata.py analyze_generality.py stats_core.py lib_driver.sh
       run_generality.sh ckpt_guard.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
declare -A VAR=([SO4]=SignedAO4_r4 [AO4]=AbsAO4_r4 [SO_bal]=SignedAO4_r4 [AO_bal]=AbsAO4_r4 [SO_bal_hi]=SignedAO4_r4
                [AO_bal_hi]=AbsAO4_r4 [Signed_r4]=Signed_r4 [CARoPEF]=CARoPEF [RoPE]=RoPE)
torus(){  # arm seed [extra args]
  local A="$1" S="$2" V="${VAR[$1]}" OUT="$R/torus/$1/p0/${VAR[$1]}_s$2" G FA="--fast-attn"; shift 2
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip torus $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip torus $A s$S (running)"; return 0; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "torus $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$R/torus/${A}_s$S.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --data-workers 3 $FA --device "cuda:$G" --output-dir "$OUT" "$@"
}
ring(){  # arm seed
  local A="$1" S="$2" OUT="$R/ring/$1_s$2" G
  [ -f "$OUT/eval.json" ] && { _drv_log "skip ring $A s$S (done)"; return 0; }
  running "$OUT" && { _drv_log "skip ring $A s$S (running)"; return 0; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "ring $A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ring_variant --arm "$A" --p-plus 0.5 --seed "$S" \
    --device "cuda:$G" --output-dir "$OUT"
}
# phase 1: every scratch run, SO4 first at each seed (seed outer, arm inner)
for S in "${SEEDS[@]}"; do for A in SO4 AO4 Signed_r4 CARoPEF RoPE; do torus "$A" "$S" --epochs 900; done; done
for S in "${SEEDS[@]}"; do for A in Signed_r4 CARoPEF CoPEF RoPE; do ring "$A" "$S"; done; done
# phase 2: constructions and continuations, per seed as SO4 finishes
for S in "${SEEDS[@]}"; do
  SRC="$R/torus/SO4/p0/SignedAO4_r4_s$S/SignedAO4_r4.pt"
  while [ ! -f "$SRC" ] || running "$R/torus/SO4/p0/SignedAO4_r4_s$S"; do sleep 60   # audit m6: wait for the process to exit
    [ -f "$SRC" ] || running "$R/torus/SO4/p0/SignedAO4_r4_s$S" || drv_fail "SO4 s$S ended without a checkpoint"
  done
  INIT="$R/torus/init/s$S"
  if [ ! -f "$INIT/SO_bal/construct.json" ] || [ ! -f "$INIT/AO_bal/construct.json" ]; then
    G=$(drv_wait_slot)
    python3 -u -m mapformer.build_torus_exist --src "$SRC" --out "$INIT" --arms AO_bal SO_bal --device "cuda:$G" \
      >> "$LOG" 2>&1 || drv_fail "build s$S"
  fi
  torus SO_bal "$S" --epochs 300 --lr 1e-4 --init-from "$INIT/SO_bal/SignedAO4_r4.pt" --data-seed-offset 1000000
  torus AO_bal "$S" --epochs 300 --lr 1e-4 --init-from "$INIT/AO_bal/AbsAO4_r4.pt" --data-seed-offset 1000000
  torus SO_bal_hi "$S" --epochs 300 --init-from "$INIT/SO_bal/SignedAO4_r4.pt" --data-seed-offset 1000000
  torus AO_bal_hi "$S" --epochs 300 --init-from "$INIT/AO_bal/AbsAO4_r4.pt" --data-seed-offset 1000000
done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=(); for S in "${SEEDS[@]}"; do
  for A in "${!VAR[@]}"; do REQ+=("$R/torus/$A/p0/${VAR[$A]}_s$S/${VAR[$A]}.pt"); done
  for A in Signed_r4 CARoPEF CoPEF RoPE; do REQ+=("$R/ring/${A}_s$S/eval.json" "$R/ring/${A}_s$S/$A.pt"); done
  for A in SO_bal AO_bal; do REQ+=("$R/torus/init/s$S/$A/construct.json"); done
done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 152 ] || drv_fail "expected 152 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the evaluation"
for A in "${!VAR[@]}"; do
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/torus/$A" --variants "${VAR[$A]}" --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 1024 2048 --n-trials 100 --device cuda:0 --title "GENERALITY torus $A" --out "$REPO/GENERALITY_TORUS_$A.md" \
    >> "$LOG" 2>&1 || drv_fail "eval $A"
  python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/torus/$A" --variants "${VAR[$A]}" --seeds "${SEEDS[@]}" --lengths 1024 \
    --n-trials 100 --device cuda:0 --check-json "$REPO/GENERALITY_TORUS_$A.json" --out "$REPO/GENERALITY_TORUS_${A}_STRATA.json" \
    >> "$LOG" 2>&1 || drv_fail "strata $A"
done
for A in AO4 SO4 CARoPEF RoPE Signed_r4; do   # declared secondary: dropout-scale re-score (one layer), T = 1024
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_noise_refine --runs-dir "$R/torus/$A" --variants "${VAR[$A]}" \
    --noises 0.0 --seeds "${SEEDS[@]}" --lengths 1024 --n-trials 100 --device cuda:0 --title "GENERALITY torus $A re-scored" \
    --out "$REPO/GENERALITY_TORUS_${A}_RS.md" >> "$LOG" 2>&1 || drv_fail "rescore $A"
done
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_generality > "$REPO/GENERALITY_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.generality_done" "$REPO/GENERALITY_ANALYSIS.txt" "$REPO/GENERALITY_VERDICTS.json"
