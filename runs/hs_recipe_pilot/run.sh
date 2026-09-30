#!/usr/bin/env bash
# Hidden-state step recipe pilot (CTXSTEP_PILOT3.md, "Before a registered batch"). Far cue, both sides,
# seeds 1 and 5 (text-world seeds whose context-free model solved cleanly). Not reused.
#   cold: HS from scratch, 1800 epochs          warm: HS warm-started from runs/textworld Vanilla_r4_L1_s<seed>, 900 epochs
set -uo pipefail
REPO=/home/prashr/mapformer; LOG=$REPO/runs/hs_recipe_pilot/driver.log
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING=15; DRV_MODULE_RE="mapformer[.]train_(variant|ctxstep|ctxstep2|ctxstep3|textworld)"
drv_lock "$REPO/.run_hs_recipe_pilot.lock" || exit 1
cd /home/prashr; R=$REPO/runs/hs_recipe_pilot
for S in 1 5; do for CUE in lead trail; do for REC in cold warm; do
  OUT="$R/${CUE}_far_${REC}_s${S}"; [ -f "$OUT/eval.json" ] && continue; mkdir -p "$OUT"
  if [ $REC = cold ]; then EX="--epochs 1800"; else EX="--epochs 900 --warm-cf $REPO/runs/textworld/p0/Vanilla_r4_L1_s${S}/Vanilla_r4.pt"; fi
  G=$(drv_wait_slot); echo "$(date +%H:%M:%S) $CUE far $REC s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_ctxstep3 --variant HS --n-layers 2 \
    --cue $CUE --dist far --seed $S --n-steps 2048 --batch-size 8 $EX --device "cuda:$G" --output-dir "$OUT"
done; done; done
drv_wait_dir "$R/"; touch "$R/.done"
