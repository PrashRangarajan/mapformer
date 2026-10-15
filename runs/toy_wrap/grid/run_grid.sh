#!/usr/bin/env bash
# exploratory toy grid (CPU), 8 seeds x configs, 8 at a time
cd /home/prashr
O=/home/prashr/mapformer/runs/toy_wrap/grid
CFGS=("signed random" "abs random" "abs random --init-scale 3" "signed low" "abs low" "signed high" "abs high" "abs high --no-omega-clamp" "abs band" "abs offband")
for S in 10 11 12 13 14 15 16 17; do
  for c in "${CFGS[@]}"; do
    set -- $c; M=$1; I=$2; shift 2
    N="${M}_${I}$(echo "$@" | tr ' -' '__')_s$S"
    [ -f "$O/$N.json" ] && continue
    while [ "$(jobs -r | wc -l)" -ge 8 ]; do sleep 2; done
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python3 -m mapformer.toy_wrap --mode $M --init $I "$@" --seed $S --steps 3000 --log-every 1000 --out "$O/$N.json" > "$O/$N.log" 2>&1 &
  done
done
wait; touch $O/.done
