#!/usr/bin/env bash
# instrumented AdamW at every grid rate (incl. extension rates), CPU, seed 1580, 20 epochs x 5 batches
D=/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/lrsweep_check
cd /home/prashr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=/home/prashr
for L in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1; do for A in SO AO; do
  python3 $D/wrap_gauge.py $D/lr/${A}_$L.json -- --arm $A --deltas=1,-1 --seed 1580 --lr $L --weight-decay 0.05 \
    --device cpu --output-dir $D/lr/${A}_$L --epochs 20 --n-batches 5 --n-trials 20 > $D/lr/${A}_$L.log 2>&1 &
done; done; wait
# bitwise: LRSWEEP 1e-3 cell vs ROBUST def cell command line (identical flags), seed 1581, 3 epochs x 4 batches
for A in SO AO; do
  python3 -m mapformer.train_gauge --arm $A --deltas=1,-1 --seed 1581 --lr 1e-3 --weight-decay 0.05 --device cpu \
    --output-dir $D/bit/lrsweep_$A --epochs 3 --n-batches 4 --n-trials 20 > $D/bit/lrsweep_$A.log 2>&1 &
  python3 -m mapformer.train_gauge --arm $A --deltas=1,-1 --seed 1581 --lr 1e-3 --weight-decay 0.05 --device cpu \
    --output-dir $D/bit/robust_$A --epochs 3 --n-batches 4 --n-trials 20 > $D/bit/robust_$A.log 2>&1 &
  python3 -m mapformer.train_gauge --arm $A --deltas=1,-1 --seed 1581 --lr 0.001 --device cpu \
    --output-dir $D/bit/default_$A --epochs 3 --n-batches 4 --n-trials 20 > $D/bit/default_$A.log 2>&1 &
done; wait
# divergence probe: absurd rates, seed 1582, 6 epochs x 5 batches
for L in 1e2 1e3 1e4; do
  python3 -m mapformer.train_gauge --arm AO --deltas=1,-1 --seed 1582 --lr $L --weight-decay 0.05 --device cpu \
    --output-dir $D/div/AO_$L --epochs 6 --n-batches 5 --n-trials 20 > $D/div/AO_$L.log 2>&1 &
  python3 -m mapformer.train_gauge --arm SO --deltas=1,-1 --seed 1582 --lr $L --weight-decay 0.05 --device cpu \
    --output-dir $D/div/SO_$L --epochs 6 --n-batches 5 --n-trials 20 > $D/div/SO_$L.log 2>&1 &
done; wait
touch $D/lr_check.done
