#!/usr/bin/env bash
# New-object transfer pilot: 6 arms x seed 100 (outside the registered seeds), full recipe. Not reused.
set -uo pipefail
REPO=/home/prashr/mapformer; LOG=$REPO/runs/newobj_pilot/driver.log
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING=15
drv_lock "$REPO/.run_newobj_pilot.lock" || exit 1
cd /home/prashr; R=$REPO/runs/newobj_pilot
for ARM in MapWM MapPoPE MapEM PosOnly RoPE PoPE; do
  OUT="$R/${ARM}_s100"; [ -f "$OUT/eval.json" ] && continue; mkdir -p "$OUT"
  G=$(drv_wait_slot); echo "$(date +%H:%M:%S) $ARM -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$OUT.log" python3 -u -m mapformer.train_newobj --variant $ARM --seed 100 \
    --epochs 900 --n-steps 1024 --batch-size 16 --device "cuda:$G" --output-dir "$OUT"
done
drv_wait_dir "$R/"; touch "$R/.done"
