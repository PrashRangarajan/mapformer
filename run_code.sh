#!/bin/bash
# Code-modelling batch. Pre-registration: CODE_PREREG.md. Gates: CODE_GATES.md.
#
# Seed OUTER, variant INNER: a full four-arm table lands after wave 1 instead of
# one arm at a time.
#
# The wait loop uses `ps -o comm=` and NOT `pgrep -f`. pgrep -f matches any shell
# whose command line mentions the pattern -- including the tool call that launched
# this and every later one that discusses it. A previous script sat in its wait
# loop for two hours against zero running jobs for exactly that reason. `comm` is
# the executable name, so shells cannot match however they quote things.
set -u
REPO=/home/prashr/mapformer
OUT=$REPO/runs/code
mkdir -p "$OUT"
cd /home/prashr

ARMS=(RoPE PoPE-Flat Vanilla MapPoPE-Flat)
MAXPG=4

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_hourglass_enwik8/' | wc -l; }

launch(){
  local arm=$1 seed=$2 gpu=$3
  local tag="_s${seed}"
  if [ -f "$OUT/${arm}${tag}.json" ]; then echo "skip $arm seed$seed (done)"; return; fi
  echo "launch $arm seed=$seed on cuda:$gpu  $(date +%H:%M:%S)"
  OMP_NUM_THREADS=4 setsid nohup python3 -m mapformer.train_hourglass_enwik8 \
    --model "$arm" --device "cuda:$gpu" --seed "$seed" \
    --iters 36000 --eval-every 1000 --save-ckpt \
    --bottleneck-r 4 --seq-len 512 --batch-size 16 --dim 512 --n-layers 9 --lr 2e-4 \
    --data $REPO/data/code_train.bin --data-val $REPO/data/code_val.bin \
    --out "$OUT" --tag "$tag" \
    >> "$OUT/${arm}${tag}.log" 2>&1 &
  sleep 20
}

i=0
for seed in 0 1 2; do
  for arm in "${ARMS[@]}"; do
    while [ "$(busy)" -ge "$MAXPG" ]; do sleep 60; done
    launch "$arm" "$seed" $((i % 2))
    i=$((i+1))
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 60; done

# Verify the artifacts exist. `wait` returns regardless of child success and a
# script once touched its completion marker after every arm had died.
missing=0
for seed in 0 1 2; do for arm in "${ARMS[@]}"; do
  [ -f "$OUT/${arm}_s${seed}.json" ] || { echo "MISSING ${arm}_s${seed}"; missing=$((missing+1)); }
done; done
echo "missing=$missing"
[ "$missing" -eq 0 ] && touch "$OUT/.code_done"
echo "batch finished $(date)"
