#!/usr/bin/env bash
# RING_DIAL_PREREG.md batch (+ Amendment 1): Signed_r4 / Abs_r4 x p_plus {0.5, 0.75, 0.9} x seeds 30-37 on the ring; GPUs from
# DRV_GPUS. Waits for CLOCK_SIDE's done marker first (shared slot budget on GPU 0, audit minor).
#   cd /home/prashr/mapformer && DRV_GPUS=0 setsid nohup bash run_ring_dial.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/ring_dial.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27); ring jobs peak ~1.25 GB
export DRV_GPUS="${DRV_GPUS:-0}"
drv_lock "$REPO/.run_ring_dial.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/ring_dial/p0"; mkdir -p "$R"
GUARD=(train_ring_dial.py train_cancel.py environment_cancel.py environment_nd.py train_variant.py model.py model_sign.py
       model_baseline_rope.py train.py analyze_ring_dial.py stats_core.py lib_driver.sh run_ring_dial.sh data_parallel.py
       environment.py rescore_hook.py)
echo "start $(date)" >> "$LOG"
while [ ! -f "$REPO/.clock_side_done" ]; do sleep 120; done; _drv_log "CLOCK_SIDE done; starting"
drv_md5_guard "$REPO/runs/ring_dial" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
for S in 30 31 32 33 34 35 36 37; do for P in 0.5 0.75 0.9; do for A in Signed_r4 Abs_r4; do
  OUT="$R/${A}_p${P}_s${S}"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $A p$P s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $A p$P s$S (running)"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$A p$P s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ring_dial --arm "$A" --p-plus "$P" --seed "$S" \
    --device "cuda:$G" --output-dir "$OUT"
done; done; done
sleep 60
while ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_ring_dial/ && /runs\/ring_dial\/p0/' | grep -q .; do sleep 60; done
REQ=(); for S in 30 31 32 33 34 35 36 37; do for P in 0.5 0.75 0.9; do for A in Signed_r4 Abs_r4; do REQ+=("$R/${A}_p${P}_s${S}/eval.json" "$R/${A}_p${P}_s${S}/${A}.pt"); done; done; done
drv_require "${REQ[@]}" || drv_fail "missing runs"
drv_md5_guard "$REPO/runs/ring_dial" "${GUARD[@]}" || drv_fail "code changed before the analysis"
[ "${#REQ[@]}" -eq 96 ] || drv_fail "expected 96 artifacts"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_ring_dial > "$REPO/RING_DIAL_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.ring_dial_done" "$REPO/RING_DIAL_ANALYSIS.txt" "$REPO/RING_DIAL_VERDICTS.json"
