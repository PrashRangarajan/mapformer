#!/bin/bash
# ABLATE_PREREG.md: PoPE component ablation on the code corpus, trained at 512,
# evaluated at 2048. All four arms in one batch (PoPE-Full is a fresh control).
set -u
REPO=/home/prashr/mapformer
OUT=$REPO/runs/code_ablate
mkdir -p "$OUT"
cd /home/prashr
ARMS=(PoPE-Full PoPE-NoSigma PoPE-ReLU PoPE-NoDelta)
MAXPG=${MAXPG:-4}
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_hourglass_enwik8/ && /runs\/code_ablate/' | wc -l; }
i=0
for seed in 0 1 2; do
  for arm in "${ARMS[@]}"; do
    [ -f "$OUT/${arm}_s${seed}.json" ] && continue
    while [ "$(busy)" -ge "$MAXPG" ]; do sleep 60; done
    echo "launch $arm seed=$seed cuda:$((i%2)) $(date +%H:%M:%S)"
    OMP_NUM_THREADS=4 setsid nohup python3 -u -m mapformer.train_hourglass_enwik8 \
      --model "$arm" --device "cuda:$((i%2))" --seed "$seed" \
      --iters 36000 --eval-every 1000 --save-ckpt \
      --bottleneck-r 4 --seq-len 512 --batch-size 16 --dim 512 --n-layers 9 --lr 2e-4 \
      --data $REPO/data/code_train.bin --data-val $REPO/data/code_val.bin \
      --out "$OUT" --tag "_s${seed}" >> "$OUT/${arm}_s${seed}.log" 2>&1 &
    i=$((i+1)); sleep 20
  done
done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
missing=0
for seed in 0 1 2; do for arm in "${ARMS[@]}"; do
  [ -f "$OUT/${arm}_s${seed}.json" ] || { echo "MISSING ${arm}_s${seed}"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "ablation batch finished $(date)"
