#!/usr/bin/env bash
# RANK_NOWRAP pilot: timing at 8 concurrent (4/GPU) on seeds 100-101, 40 epochs; plus the bitwise repro (15 epochs).
REPO=/home/prashr/mapformer; P=$REPO/runs/rank_nowrap_pilot; mkdir -p $P/repro
cd /home/prashr
OMP_NUM_THREADS=2 setsid nohup python3 -u $REPO/docs/audits/2026-10-06/rank_nowrap_repro.py 15 $P/repro/repro_losses.json \
  --variant Vanilla_r2ph_om32 --env nd --n-dims 2 --grid-size 32 --seed 0 --epochs 900 --lr 1e-3 --n-batches 98 \
  --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
  --data-workers 3 --device cuda:0 --output-dir $P/repro/out --save-full-state > $P/repro/repro.log 2>&1 &
i=0
for S in 100 101; do for N in 32 256; do for V in Vanilla_r2ph_om32 Vanilla_r3ph_om32; do
  G=$((i % 2)); i=$((i+1)); OUT=$P/N$N/D2/${V}_s$S; mkdir -p $OUT
  OMP_NUM_THREADS=2 setsid nohup python3 -u -m mapformer.train_rank_nowrap --variant $V --env nd --n-dims 2 --grid-size $N \
    --seed $S --epochs 40 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 \
    --n-landmarks 0 --schedule cosine --data-workers 3 --device cuda:$G --output-dir $OUT --save-full-state > $P/N${N}_${V}_s$S.log 2>&1 &
done; done; done
echo "pilot launched $(date)" > $P/pilot_started.txt
