#!/usr/bin/env bash
# TinyStories word-level pilot (TINYSTORIES_PILOT.md): MapWM (Vanilla, rank 4) and RoPE, seeds 0-2, on the GPUs listed
# in DRV_GPUS (default 0). Seed outer, arm inner. Launch:
#   cd /home/prashr/mapformer && DRV_GPUS=0 setsid nohup bash run_tinystories.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/tinystories.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=4000; DRV_SPACING=20     # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0}"
drv_lock "$REPO/.tinystories.lock" || exit 0
cd "$REPO/.."
R="$REPO/runs/tinystories/p0"; mkdir -p "$R"
echo "start $(date)" >> "$LOG"
_drv_log "code version at launch: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS"
for S in 0 1 2; do for M in Vanilla RoPE; do
  [ -f "$R/${M}_s${S}.json" ] && { _drv_log "skip ${M}_s${S} (done)"; continue; }
  if [ -n "$(ps -u "$USER" -o comm=,args= | awk -v m="--model $M --seed $S " '$1=="python3" && /mapformer[.]train_tinystories/ && index($0" ", m)')" ]; then
    _drv_log "skip ${M}_s${S} (already running)"; continue
  fi
  G=$(drv_wait_slot)
  _drv_log "${M}_s${S} -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$R/${M}_s${S}.log" python3 -u -m mapformer.train_tinystories --model "$M" --seed "$S" \
    --device "cuda:$G" --out "$R"
done; done
sleep 60
while [ "$(ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer[.]train_tinystories/' | wc -l)" -gt 0 ]; do sleep 60; done
REQ=(); for S in 0 1 2; do for M in Vanilla RoPE; do REQ+=("$R/${M}_s${S}.json" "$R/${M}_s${S}.best.pt"); done; done
drv_require "${REQ[@]}" || drv_fail "missing artifacts"
drv_done "$REPO/.tinystories_done" "${REQ[@]}"
