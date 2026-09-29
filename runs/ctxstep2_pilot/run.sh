#!/usr/bin/env bash
# Context-step pilot 2 (CONTEXT_STEP_DESIGN.md "Revision"): CF, SR, CG (1 layer), HS (2) x {lead, trail}
# x 2 seeds. Starts after pilot 1 (runs/ctxstep_pilot/.done). Not reused.
set -uo pipefail
REPO=/home/prashr/mapformer; LOG=$REPO/runs/ctxstep2_pilot/driver.log
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-3}"; DRV_SPACING=15; DRV_MODULE_RE="mapformer[.]train_(variant|ctxstep|ctxstep2)"
drv_lock "$REPO/.run_ctxstep2_pilot.lock" || exit 1
while [ ! -e "$REPO/runs/ctxstep_pilot/.done" ]; do sleep 60; done
cd /home/prashr; R=$REPO/runs/ctxstep2_pilot
for S in 0 1; do for CUE in lead trail; do for c in "CF 1" "SR 1" "CG 1" "HS 2"; do set -- $c
  OUT="$R/${CUE}_${1}_L${2}_s${S}"; [ -f "$OUT/eval.json" ] && continue; mkdir -p "$OUT"
  G=$(drv_wait_slot); echo "$(date +%H:%M:%S) $CUE $1 L$2 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_ctxstep2 --variant "$1" --n-layers "$2" \
    --cue "$CUE" --seed "$S" --device "cuda:$G" --output-dir "$OUT"
done; done; done
drv_wait_dir "$R/"; touch "$R/.done"
