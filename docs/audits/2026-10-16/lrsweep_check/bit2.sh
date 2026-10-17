D=/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/lrsweep_check
cd /home/prashr; export CUDA_VISIBLE_DEVICES=""
for i in 1 2; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m mapformer.train_gauge --arm SO --deltas=1,-1 --seed 1583 --lr 1e-3 --weight-decay 0.05 --device cpu \
    --output-dir $D/bit2/omp1_$i --epochs 3 --n-batches 4 --n-trials 20 --data-workers 3 > $D/bit2/omp1_$i.log 2>&1
done
for i in 1 2; do
  OMP_NUM_THREADS=2 python3 -m mapformer.train_gauge --arm SO --deltas=1,-1 --seed 1583 --lr 1e-3 --weight-decay 0.05 --device cpu \
    --output-dir $D/bit2/omp2_$i --epochs 3 --n-batches 4 --n-trials 20 > $D/bit2/omp2_$i.log 2>&1
done
touch $D/bit2.done
