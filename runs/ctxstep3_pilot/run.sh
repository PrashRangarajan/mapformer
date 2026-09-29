#!/usr/bin/env bash
# Context-step pilot 3 (cue distance; CONTEXT_STEP_DESIGN.md, second revision). 1 seed. Not reused.
# far x {lead, trail} x {CF, SR, CG, HS}; near x {lead, trail} x {SR, CG}. T=2048 words, batch 8, 900 epochs.
set -uo pipefail
REPO=/home/prashr/mapformer; LOG=$REPO/runs/ctxstep3_pilot/driver.log
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-3}"; DRV_SPACING=15; DRV_MODULE_RE="mapformer[.]train_(variant|ctxstep|ctxstep2|ctxstep3|textworld)"
drv_lock "$REPO/.run_ctxstep3_pilot.lock" || exit 1
cd /home/prashr; R=$REPO/runs/ctxstep3_pilot
JOBS=(); for CUE in lead trail; do for c in "CF 1" "SR 1" "CG 1" "HS 2"; do JOBS+=("far $CUE $c"); done; done
for CUE in lead trail; do for c in "SR 1" "CG 1"; do JOBS+=("near $CUE $c"); done; done
for j in "${JOBS[@]}"; do set -- $j
  OUT="$R/${2}_${1}_${3}_L${4}_s0"; [ -f "$OUT/eval.json" ] && continue; mkdir -p "$OUT"
  G=$(drv_wait_slot); echo "$(date +%H:%M:%S) $j -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_ctxstep3 --variant "$3" --n-layers "$4" \
    --cue "$2" --dist "$1" --seed 0 --n-steps 2048 --batch-size 8 --device "cuda:$G" --output-dir "$OUT"
done
drv_wait_dir "$R/"; touch "$R/.done"
