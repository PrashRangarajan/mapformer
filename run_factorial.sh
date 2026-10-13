#!/usr/bin/env bash
# FACTORIAL_PREREG.md: RoPE at 1 layer / 4 layers / looped x4 and MapWM (Vanilla_r4) at 1 layer on the torus, T = 1024,
# 900 epochs, --fast-attn, seeds 500-503; GPUs from DRV_GPUS (default 0). Launch:
#   cd /home/prashr/mapformer && DRV_GPUS=0 setsid nohup bash run_factorial.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/factorial.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=4; DRV_MINFREE=4000; DRV_SPACING=30     # assigned after sourcing (rule 27); ~2 GB per job with --fast-attn
export DRV_GPUS="${DRV_GPUS:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_factorial.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/factorial"; mkdir -p "$R/L1/p0" "$R/L4/p0" "$R/LOOP/p0"
GUARD=(train_variant.py train.py model.py model_rank.py model_looped.py model_baseline_rope.py environment.py data_parallel.py
       eval_noise_refine.py eval_rank_strata.py ckpt_guard.py analyze_factorial.py stats_core.py lie_groups.py lib_driver.sh
       run_factorial.sh)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
cell(){ local ROOT=$1 V=$2 NL=$3 S=$4 OUT="$R/$1/p0/${2}_s${4}" G
  [ -f "$OUT/${V}.pt" ] && { _drv_log "skip $ROOT $V s$S (done)"; return; }
  running "$OUT" && { _drv_log "skip $ROOT $V s$S (running)"; return; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$ROOT $V L$NL s$S -> cuda:$G"
  OMP_NUM_THREADS=3 drv_launch "$R/$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
    --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers "$NL" --n-heads 2 --d-model 128 \
    --n-landmarks 0 --schedule cosine --data-workers 3 --fast-attn --device "cuda:$G" --output-dir "$OUT"; }
for S in 500 501 502 503; do        # seed outer, cell inner; deep cells first within a seed
  cell L4 RoPE 4 "$S"; cell LOOP RoPELooped 1 "$S"; cell L1 RoPE 1 "$S"; cell L1 Vanilla_r4 1 "$S"
done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_variant/ && /runs\/factorial\//' | grep -q .; do sleep 120; done
REQ=(); for S in 500 501 502 503; do REQ+=("$R/L4/p0/RoPE_s$S/RoPE.pt" "$R/LOOP/p0/RoPELooped_s$S/RoPELooped.pt" \
  "$R/L1/p0/RoPE_s$S/RoPE.pt" "$R/L1/p0/Vanilla_r4_s$S/Vanilla_r4.pt"); done
[ "${#REQ[@]}" -eq 16 ] || drv_fail "expected 16 checkpoints"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
G=$(drv_wait_slot)
for spec in "L1:RoPE Vanilla_r4" "L4:RoPE" "LOOP:RoPELooped"; do ROOT=${spec%%:*}; VS=${spec#*:}
  python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds 500 501 502 503 \
    --lengths 512 1024 2048 --n-trials 100 --device "cuda:$G" --title "Factorial $ROOT" --out "$REPO/FACTORIAL_${ROOT}.md" \
    >> "$LOG" 2>&1 || drv_fail "eval $ROOT"
  python3 -u -m mapformer.eval_rank_strata --runs-dir "$R/$ROOT" --variants $VS --seeds 500 501 502 503 --lengths 1024 \
    --n-trials 100 --device "cuda:$G" --check-json "$REPO/FACTORIAL_${ROOT}.json" --out "$REPO/FACTORIAL_${ROOT}_STRATA.json" \
    >> "$LOG" 2>&1 || drv_fail "strata $ROOT"
  for S in 500 501 502 503; do        # Amendment 1 (D3): each run on its OWN training map (env seed = run seed), secondary
    python3 -u -m mapformer.eval_noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds "$S" --env-seed "$S" \
      --lengths 1024 --n-trials 100 --device "cuda:$G" --title "Factorial $ROOT own map s$S" \
      --out "$REPO/FACTORIAL_${ROOT}_OWN_s${S}.md" >> "$LOG" 2>&1 || _drv_log "own-map eval $ROOT s$S failed (secondary)"
  done
done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
python3 -u -m mapformer.analyze_factorial > "$REPO/FACTORIAL_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.factorial_done" "$REPO/FACTORIAL_ANALYSIS.txt" "$REPO/FACTORIAL_VERDICTS.json"
