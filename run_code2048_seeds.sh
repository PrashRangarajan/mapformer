#!/bin/bash
# CODE_PREREG Amendment 3: seeds 1-2 for all four arms at 2048, powering the
# C1 retraction. Seed 0 already exists in runs/code2048. 13.5-14.7 GiB each, so
# one per 24 GiB card.
set -u

# SINGLE-INSTANCE GUARD. Killing a supervisor does NOT kill its children: `kill`
# takes the parent and the launcher keeps running, so the next supervisor starts
# a SECOND copy. On 2026-09-22 that produced 16 launches for 12 runs in
# run_ablate.sh -- both copies passed the same "does the .json exist yet" test and
# both launched. A two-sided guard needs both sides; this is the other side.
exec 9>"/tmp/.mapformer_$(basename "$0" .sh).lock"
flock -n 9 || { echo "another instance of $(basename "$0") is already running -- exiting"; exit 0; }
REPO=/home/prashr/mapformer
OUT=$REPO/runs/code2048
cd /home/prashr
ARMS=(RoPE PoPE-Flat Vanilla MapPoPE-Flat)
MAXPG=${MAXPG:-2}
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_hourglass_enwik8/ && /runs\/code2048/' | wc -l; }
i=0
for seed in 1 2; do
  for arm in "${ARMS[@]}"; do
    [ -f "$OUT/${arm}_s${seed}.json" ] && continue
    while [ "$(busy)" -ge "$MAXPG" ]; do sleep 60; done
    echo "launch $arm seed=$seed cuda:$((i%2)) $(date +%H:%M:%S)"
    OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_hourglass_enwik8 \
      --model "$arm" --device "cuda:$((i%2))" --seed "$seed" \
      --iters 36000 --eval-every 1000 --save-ckpt \
      --bottleneck-r 4 --seq-len 2048 --batch-size 4 --dim 512 --n-layers 9 --lr 2e-4 \
      --data $REPO/data/code_train.bin --data-val $REPO/data/code_val.bin \
      --out "$OUT" --tag "_s${seed}" >> "$OUT/${arm}_s${seed}.log" 2>&1 &
    i=$((i+1)); sleep 25
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
missing=0
for seed in 0 1 2; do for arm in "${ARMS[@]}"; do
  [ -f "$OUT/${arm}_s${seed}.json" ] || { echo "MISSING ${arm}_s${seed}"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.seeds_done"
echo "c1 seeds finished $(date)"
