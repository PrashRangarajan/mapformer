#!/usr/bin/env bash
# Clean timing of the rank config through the CLI on an otherwise idle GPU: 10 epochs x 98 batches,
# data workers 3; train() prints the wall time of epochs 5 and 10.
set -u
O="$1"; mkdir -p "$O"; cd /home/prashr
RC="--variant Vanilla --seed 0 --n-steps 1024 --batch-size 16 --data-workers 3 --schedule cosine --lr 1e-3 --n-layers 1 --n-heads 2 --n-landmarks 0 --device cuda:0 --epochs 10 --n-batches 98"
for nm in explicit fast fast_det; do
  case $nm in explicit) X="";; fast) X="--fast-attn";; fast_det) X="--fast-attn --deterministic";; esac
  OMP_NUM_THREADS=4 python3 -u -m mapformer.train_variant $RC $X --output-dir $O/$nm > $O/$nm.log 2>&1
  echo "$nm: $(grep -h 'Epoch' $O/$nm.log | sed 's/  */ /g' | tr '\n' ';')  peak-mem n/a" >> $O/summary.txt
done
echo DONE >> $O/summary.txt
