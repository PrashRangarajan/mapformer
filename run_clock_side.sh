#!/usr/bin/env bash
# CLOCK_SIDE_PREREG.md batch: Signed_r4, CARoPE_r4, Pos_r4, RoPE x seeds 200-207 on k-back recency at the calibrated
# (K_MAX, T); GPUs from DRV_GPUS (default 0). Launch:
#   cd /home/prashr/mapformer && DRV_GPUS=0 setsid nohup bash run_clock_side.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/clock_side.log"
source "$REPO/lib_driver.sh"
K_MAX=128; T=1024                     # set by Amendment 1 (calibration)
DRV_MAXPG=4; DRV_MINFREE=4000; DRV_SPACING=20     # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0}"
drv_lock "$REPO/.run_clock_side.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/clock_side/p0"; mkdir -p "$R"
GUARD=(train_recency.py environment_recency.py train_variant.py model.py model_sign.py model_baseline_rope.py train.py
       analyze_clock_side.py stats_core.py lib_driver.sh run_clock_side.sh)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$REPO/runs/clock_side" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); K_MAX $K_MAX T $T GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in 200 201 202 203 204 205 206 207; do for V in Signed_r4 CARoPE_r4 Pos_r4 RoPE; do
  OUT="$R/${V}_s${S}"
  [ -f "$OUT/${V}_recency.json" ] && { _drv_log "skip $V s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $V s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$V s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_recency --variant "$V" --seed "$S" --k-max "$K_MAX" \
    --T "$T" --eval-T "$T" $((2 * T)) --epochs 300 --n-batches 48 --batch-size 16 --schedule cosine --lr 1e-3 \
    --n-layers 1 --d-model 128 --n-heads 2 --fast-attn --device "cuda:$G" --output-dir "$OUT"
done; done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_recency/ && /runs\/clock_side\/p0/' | grep -q .; do sleep 60; done
REQ=(); for S in 200 201 202 203 204 205 206 207; do for V in Signed_r4 CARoPE_r4 Pos_r4 RoPE; do REQ+=("$R/${V}_s${S}/${V}_recency.json" "$R/${V}_s${S}/${V}_recency.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
drv_md5_guard "$REPO/runs/clock_side" "${GUARD[@]}" || drv_fail "code changed before the analysis"
python3 -u -m mapformer.analyze_clock_side > "$REPO/CLOCK_SIDE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.clock_side_done" "$REPO/CLOCK_SIDE_ANALYSIS.txt" "$REPO/CLOCK_SIDE_VERDICTS.json"
