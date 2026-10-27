#!/bin/bash
# Memory Maze pilots WAVE 3 on the WORKSTATION (post-blind-weighted loss, --loss-steps both) (MEMORY_MAZE_PREREG.md, Pilots). The batch itself runs on the cluster.
# Data (never committed, outside the repo): /home/prashr/mm_data/zips/15x15/{eval,train-part0}.zip, arrays in
# /home/prashr/mm_data/arrays, pilot runs in /home/prashr/mm_data/pilots. Launch under setsid nohup:
#   setsid nohup bash /home/prashr/mapformer/run_mm_pilot3.sh > /home/prashr/mm_data/pilot3_driver.out 2>&1 < /dev/null &
set -uo pipefail
REPO=/home/prashr/mapformer
MMD=/home/prashr/mm_data
LOG="$MMD/pilot3_driver.log"
source "$REPO/lib_driver.sh"
drv_lock "$REPO/.run_mm_pilot3.lock" || exit 1
DRV_MAXPG=3
DRV_SPACING=20
cd /home/prashr || exit 1

[ -f "$MMD/arrays/PREP.done" ] || drv_fail "arrays not prepared"

# pilots wave 3 (outside seeds 3693-3694), up to three per GPU
P="--pilot --val-every 25 --train-range 0 1500 --epochs 600 --loss-steps both"
TASKS=(
"P_E 3693 - $P"
"R1 3693 - $P --no-probe"
"P_odo 3693 - $P --no-probe"
"R4 3693 - $P --no-probe"
"GRU 3693 - $P --no-probe"
"GPS 3693 - $P --no-probe"
"P_raw 3693 - $P --no-probe"
"P_E 3693 installfrozen $P --no-probe --install frozen"
"P_E 3693 installtrainable $P --no-probe --install trainable"
"P_E 3694 - $P --no-probe"
"R1 3694 - $P --no-probe"
)
OUTS=()
for T in "${TASKS[@]}"; do
  read -r ARM SEED TAG EXTRA <<< "$T"
  OUT="$MMD/pilots3/${ARM}_s${SEED}"; [ "$TAG" != "-" ] && OUT="${OUT}_${TAG}"
  OUTS+=("$OUT/results.json")
  [ -f "$OUT/DONE" ] && continue
  G=$(drv_wait_slot)
  mkdir -p "$OUT"
  # shellcheck disable=SC2086
  drv_launch "$OUT/stdout.log" python3 -u -m mapformer.train_mm --arm "$ARM" --seed "$SEED" --data "$MMD/arrays" \
    --out "$OUT" --device "cuda:$G" $EXTRA
done
drv_wait_dir "$MMD/pilots3/"
drv_require "${OUTS[@]}" || drv_fail "missing pilot results"
python3 -m mapformer.analyze_mm_pilot --root "$MMD/pilots3" --json "$MMD/pilots3/PILOT_SUMMARY.json" > "$MMD/pilots3/PILOT_SUMMARY.txt" 2>&1
drv_done "$MMD/.mm_pilot3_done" "$MMD/pilots3/PILOT_SUMMARY.json"
