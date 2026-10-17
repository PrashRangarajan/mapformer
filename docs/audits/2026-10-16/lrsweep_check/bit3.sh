D=/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/lrsweep_check
cd /home/prashr; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for A in SO AO; do
  # LRSWEEP run_lrsweep.sh ring() line at L=1e-3 / ROBUST run_robust.sh ring() line at C=def (LR 1e-3, WD 0.05)
  python3 -u -m mapformer.train_gauge --arm "$A" --deltas=1,-1 --seed 1581 --lr 1e-3 --weight-decay 0.05 --device cpu \
    --output-dir $D/bit3/lrsweep_$A --epochs 3 --n-batches 4 --n-trials 20 > $D/bit3/lrsweep_$A.log 2>&1 &
  python3 -u -m mapformer.train_gauge --arm "$A" --deltas=1,-1 --seed 1581 --lr "1e-3" --weight-decay "0.05" --device cpu \
    --output-dir $D/bit3/robust_$A --epochs 3 --n-batches 4 --n-trials 20 > $D/bit3/robust_$A.log 2>&1 &
done; wait; touch $D/bit3.done
