#!/usr/bin/env bash
# Hidden-state fix pilot (CTXSTEP_HS_RECIPE.md): HSR (word step + alpha * context, alpha init 0) vs HS,
# far cue, both sides, seeds 2 and 3 (0, 1, 5 used by earlier pilots), 1800 epochs, T=2048, batch 8.
set -uo pipefail
REPO=/home/prashr/mapformer; LOG=$REPO/runs/hsr_pilot/driver.log
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING=15
drv_lock "$REPO/.run_hsr_pilot.lock" || exit 1
cd /home/prashr; R=$REPO/runs/hsr_pilot
for S in 2 3; do for CUE in lead trail; do for ARM in HSR HS; do
  OUT="$R/${CUE}_far_${ARM}_s${S}"; [ -f "$OUT/eval.json" ] && continue; mkdir -p "$OUT"
  G=$(drv_wait_slot); echo "$(date +%H:%M:%S) $CUE far $ARM s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_ctxstep3 --variant $ARM --n-layers 2 \
    --cue $CUE --dist far --seed $S --epochs 1800 --n-steps 2048 --batch-size 8 --device "cuda:$G" --output-dir "$OUT"
done; done; done
drv_wait_dir "$R/"; touch "$R/.done"
