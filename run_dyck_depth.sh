#!/bin/bash
# DYCK_DEPTH_PREREG.md: E1 depth-matched 2x2 + E2 frequency-ladder control.
# Seed outer so a full low-confidence table lands early.
set -u

# SINGLE-INSTANCE GUARD. Killing a supervisor does NOT kill its children: `kill`
# takes the parent and the launcher keeps running, so the next supervisor starts
# a SECOND copy. On 2026-09-22 that produced 16 launches for 12 runs in
# run_ablate.sh -- both copies passed the same "does the .json exist yet" test and
# both launched. A two-sided guard needs both sides; this is the other side.
exec 9>"/tmp/.mapformer_$(basename "$0" .sh).lock"
flock -n 9 || { echo "another instance of $(basename "$0") is already running -- exiting"; exit 0; }
REPO=/home/prashr/mapformer
OUT=$REPO/runs/dyck_depth
mkdir -p "$OUT"
cd /home/prashr
MAXPG=${MAXPG:-4}
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_dyck/ && /runs\/dyck_depth/' | wc -l; }
run(){   # arch layers heads ropebase seed gpu
  local arch=$1 L=$2 H=$3 RB=$4 seed=$5 gpu=$6
  local nm="${arch}-${L}L"; case "$arch" in MapWM|MapEM|MapPoPE) nm="${nm}_r2";; esac
  [ "$RB" != "-" ] && nm="${nm}_b${RB}"
  [ -f "$OUT/${nm}_s${seed}/${nm}.json" ] && return
  while [ "$(busy)" -ge "$MAXPG" ]; do sleep 20; done
  local extra=""; [ "$RB" != "-" ] && extra="--rope-base $RB"
  OMP_NUM_THREADS=2 setsid nohup python3 -m mapformer.train_dyck \
    --arch "$arch" --n-layers "$L" --n-heads "$H" --rank 2 $extra \
    --seed "$seed" --device "cuda:$gpu" --output-dir "$OUT/${nm}_s${seed}" \
    >> "$OUT/${nm}_s${seed}.log" 2>&1 &
  sleep 3
}
i=0
for seed in 0 1 2 3 4 5 6 7; do
  # E1: depth-matched 2x2, all four arms in this batch
  for arch in MapWM MapPoPE RoPE PoPE; do run "$arch" 2 2 - "$seed" $((i%2)); i=$((i+1)); done
  # E2: frequency ladder on the 1-layer index arms, incl. a fresh base-10000 control
  for rb in 32 128 10000; do
    for arch in RoPE PoPE; do run "$arch" 1 1 "$rb" "$seed" $((i%2)); i=$((i+1)); done
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 20; done
echo "dyck_depth batch finished $(date)"
touch "$OUT/.done"
