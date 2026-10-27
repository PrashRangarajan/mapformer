#!/bin/bash
# Memory Maze prep + pilots on the WORKSTATION (MEMORY_MAZE_PREREG.md, Pilots). The batch itself runs on the cluster.
# Data (never committed, outside the repo): /home/prashr/mm_data/zips/15x15/{eval,train-part0}.zip, arrays in
# /home/prashr/mm_data/arrays, pilot runs in /home/prashr/mm_data/pilots. Launch under setsid nohup:
#   setsid nohup bash /home/prashr/mapformer/run_mm_pilot.sh > /home/prashr/mm_data/pilot_driver.out 2>&1 < /dev/null &
set -uo pipefail
REPO=/home/prashr/mapformer
MMD=/home/prashr/mm_data
LOG="$MMD/pilot_driver.log"
source "$REPO/lib_driver.sh"
drv_lock "$REPO/.run_mm_pilot.lock" || exit 1
DRV_SPACING=20
cd /home/prashr || exit 1

# 1. prep (one GPU): encoder, arrays, E map, encoder quality
if [ ! -f "$MMD/arrays/PREP.done" ]; then
  _drv_log "prep start"
  CUDA_VISIBLE_DEVICES=0 python3 -u "$REPO/mm_encoder.py" prep --zips "$MMD/zips" --out "$MMD/arrays" --device cuda \
    >> "$MMD/prep.out" 2>&1
  [ -f "$MMD/arrays/PREP.done" ] || drv_fail "prep did not finish"
  _drv_log "prep done"
fi

# 2. pilots (outside seeds 3690-3691), two per GPU
P="--pilot --val-every 25 --train-range 0 1500 --epochs 800"
TASKS=(
"P_E 3690 - $P"
"R1 3690 - $P --no-probe"
"P_odo 3690 - $P --no-probe"
"GRU 3690 - $P --no-probe"
"R4 3690 - $P --no-probe"
"GPS 3690 - $P --no-probe"
"P_raw 3690 - $P --no-probe"
"P_E 3690 installfrozen $P --no-probe --install frozen"
"P_E 3690 installtrainable $P --no-probe --install trainable"
"P_E 3691 - $P --no-probe"
"R1 3691 - $P --no-probe"
)
OUTS=()
for T in "${TASKS[@]}"; do
  read -r ARM SEED TAG EXTRA <<< "$T"
  OUT="$MMD/pilots/${ARM}_s${SEED}"; [ "$TAG" != "-" ] && OUT="${OUT}_${TAG}"
  OUTS+=("$OUT/results.json")
  [ -f "$OUT/DONE" ] && continue
  G=$(drv_wait_slot)
  mkdir -p "$OUT"
  # shellcheck disable=SC2086
  drv_launch "$OUT/stdout.log" python3 -u -m mapformer.train_mm --arm "$ARM" --seed "$SEED" --data "$MMD/arrays" \
    --out "$OUT" --device "cuda:$G" $EXTRA
done
drv_wait_dir "$MMD/pilots/"
drv_require "${OUTS[@]}" || drv_fail "missing pilot results"
python3 -m mapformer.analyze_mm_pilot --root "$MMD/pilots" --json "$MMD/pilots/PILOT_SUMMARY.json" > "$MMD/pilots/PILOT_SUMMARY.txt" 2>&1
drv_done "$MMD/.mm_pilot_done" "$MMD/pilots/PILOT_SUMMARY.json"
