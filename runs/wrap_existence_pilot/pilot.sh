#!/usr/bin/env bash
# Pilot on seed 690 (outside the batch): SO and AO from scratch, then SO_cont and AO_wrap from the SO checkpoint.
cd /home/prashr
P=/home/prashr/mapformer/runs/wrap_existence_pilot
for A in SO AO; do
  OMP_NUM_THREADS=2 python3 -u -m mapformer.train_wrap_exist --arm $A --seed 690 --device cuda:0 --output-dir $P/${A}_s690 > $P/${A}_s690.log 2>&1 &
done
wait
for A in SO_cont AO_wrap; do
  OMP_NUM_THREADS=2 python3 -u -m mapformer.train_wrap_exist --arm $A --seed 690 --init $P/SO_s690/SO.pt --device cuda:0 --output-dir $P/${A}_s690 > $P/${A}_s690.log 2>&1 &
done
wait
touch $P/.pilot_done
