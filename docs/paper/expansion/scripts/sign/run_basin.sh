#!/bin/bash
# SEARCH_THEORY A3 driver (CPU): 10 jobs at a time, 2 threads each. Writes runs_basin/<arm>[_t<tau>]_s<seed>.json and .log.
cd /home/prashr/mapformer/docs/paper/expansion/scripts/sign
mkdir -p runs_basin
jobs=()
for seed in 800 801 802 803 804 805 806 807; do
  for arm in S_def S_abs S_abs_fz A_abs A_high; do jobs+=("$arm 0.5 $seed $arm"); done
done
for seed in 800 801 802 803 804 805 806 807; do
  for tau in 0.25 0.5 0.75; do jobs+=("A_seg $tau $seed A_seg_t$tau"); done
  jobs+=("S_seg 0.75 $seed S_seg_t0.75")
done
printf '%s\n' "${jobs[@]}" | xargs -P 10 -L 1 bash -c 'f=runs_basin/$3_s$2; [ -s $f.json ] || python3 step_basin.py --arm $0 --tau $1 --seed $2 --steps 2000 --log-every 200 --out $f.json > $f.log 2>&1'
touch runs_basin/ALL.done
