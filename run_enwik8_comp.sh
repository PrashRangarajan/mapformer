#!/bin/bash
# ENWIK8_COMPOSITION_PREREG.md: n=12 on the comparison actually being claimed.
# Seed OUTER so a full low-confidence table lands early and n grows uniformly.
set -u
REPO=/home/prashr/mapformer
OUT=$REPO/runs/enwik8_comp
mkdir -p "$OUT"
cd /home/prashr
ARMS=(MapPoPE-Flat PoPE-Flat Vanilla)
MAXPG=${MAXPG:-6}
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /train_hourglass_enwik8/ && /runs\/enwik8_comp/' | wc -l; }
launch(){
  local arm=$1 seed=$2 gpu=$3
  [ -f "$OUT/${arm}_s${seed}.json" ] && return
  while [ "$(busy)" -ge "$MAXPG" ]; do sleep 60; done
  echo "launch $arm seed=$seed cuda:$gpu $(date +%H:%M:%S)"
  OMP_NUM_THREADS=2 setsid nohup python3 -m mapformer.train_hourglass_enwik8 \
    --model "$arm" --device "cuda:$gpu" --seed "$seed" \
    --iters 36000 --eval-every 1000 --save-ckpt \
    --bottleneck-r 4 --seq-len 512 --batch-size 16 --dim 512 --n-layers 9 --lr 2e-4 \
    --out "$OUT" --tag "_s${seed}" >> "$OUT/${arm}_s${seed}.log" 2>&1 &
  sleep 20
}
# Power is matched per CONTRAST, not uniform across arms. MapPoPE-PoPE is the
# binding contrast and needs n=12 (effect -0.0020 at paired sd 0.0023). The
# MapPoPE-MapWM contrast has a ~3x larger effect (-0.0056) and needs n~1.3, so
# n=8 there is still 2.5x over-powered while saving four runs. Unequal n is fine
# because every contrast is paired on its own common seeds.
i=0
for seed in $(seq 0 11); do
  for arm in "${ARMS[@]}"; do
    [ "$arm" = "Vanilla" ] && [ "$seed" -ge 8 ] && continue
    launch "$arm" "$seed" $((i % 2)); i=$((i+1));
  done
  # RoPE reproduction control on seeds 0-2 only (stored: 1.3864 / 1.3837 / 1.3840)
  if [ "$seed" -lt 3 ]; then launch RoPE "$seed" $((i % 2)); i=$((i+1)); fi
done
while [ "$(busy)" -gt 0 ]; do sleep 60; done
missing=0
for seed in $(seq 0 11); do for arm in "${ARMS[@]}"; do
  [ "$arm" = "Vanilla" ] && [ "$seed" -ge 8 ] && continue
  [ -f "$OUT/${arm}_s${seed}.json" ] || { echo "MISSING ${arm}_s${seed}"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; [ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "enwik8 composition batch finished $(date)"
