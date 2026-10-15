#!/usr/bin/env bash
# ACCOUNTS T3 (CPU toy, exploratory): observation-retrieval target, 4 starts x seeds 10-17, OMP 2 (bit-deterministic at 2).
cd /home/prashr
O=/home/prashr/mapformer/runs/accounts/t3
for S in 10 11 12 13 14 15 16 17; do for c in "signed absmag obs" "abs absmag obs" "signed random obs" "abs random obs" \
                                               "signed absmag cell" "abs absmag cell"; do
  set -- $c; N="${1}_${2}_${3}_s$S"; [ -f "$O/$N.json" ] && continue
  while [ "$(jobs -r | wc -l)" -ge "${T3_PAR:-4}" ]; do sleep 2; done
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python3 -m mapformer.toy_wrap --mode $1 --init $2 --target $3 --seed $S \
    --steps 3000 --log-every 1000 --out "$O/$N.json" > "$O/$N.log" 2>&1 &
done; done
wait; touch "$O/.done"
