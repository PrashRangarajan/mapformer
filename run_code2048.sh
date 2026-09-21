#!/bin/bash
# Control for "can't you just train longer?": same corpus, same tokens per step
# (batch 4 x 2048 = 16 x 512), trained AND tested at 2048. If the arms ceiling
# here the way they did at 512, the OOD code result is a train/test mismatch
# artifact and the framing must be retracted.
#
# Peak memory is 13.5-14.7 GiB per run, so ONE per 24 GiB card, not two.
# busy() counts only THIS experiment's jobs (matched on --out) so the 512-context
# decay batch can share the GPUs without the two drivers fighting over slots.
set -u
REPO=/home/prashr/mapformer
OUT=$REPO/runs/code2048
mkdir -p "$OUT"
cd /home/prashr

ARMS=(RoPE PoPE-Flat Vanilla MapPoPE-Flat)
MAXPG=2

busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_hourglass_enwik8/ && /runs\/code2048/' | wc -l; }

i=0
for seed in 0; do
  for arm in "${ARMS[@]}"; do
    [ -f "$OUT/${arm}_s${seed}.json" ] && { echo "skip $arm s$seed"; continue; }
    while [ "$(busy)" -ge "$MAXPG" ]; do sleep 60; done
    echo "launch $arm seed=$seed on cuda:$((i % 2))  $(date +%H:%M:%S)"
    OMP_NUM_THREADS=4 setsid nohup python3 -m mapformer.train_hourglass_enwik8 \
      --model "$arm" --device "cuda:$((i % 2))" --seed "$seed" \
      --iters 36000 --eval-every 1000 --save-ckpt \
      --bottleneck-r 4 --seq-len 2048 --batch-size 4 --dim 512 --n-layers 9 --lr 2e-4 \
      --data $REPO/data/code_train.bin --data-val $REPO/data/code_val.bin \
      --out "$OUT" --tag "_s${seed}" >> "$OUT/${arm}_s${seed}.log" 2>&1 &
    i=$((i+1)); sleep 25
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
missing=0
for arm in "${ARMS[@]}"; do [ -f "$OUT/${arm}_s0.json" ] || { echo "MISSING $arm"; missing=$((missing+1)); }; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "2048 batch finished $(date)"
