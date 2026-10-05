#!/usr/bin/env bash
# Re-score of rank_nd and rank_wrap (eval_nd, registered streams) under rescore_hook: scale none and auto, T=1024 only.
set -uo pipefail
REPO=/home/prashr/mapformer; O=$REPO/runs_rescore; cd /home/prashr
for sc in none auto; do
  python3 -m mapformer.rescore_hook --scale $sc -- mapformer.eval_nd --runs-dir $REPO/runs/rank_nd --configs "2:32:Vanilla_r2ph,Vanilla_r3ph" \
    "3:10:Vanilla_r3ph,Vanilla_r4ph" --lengths 1024 --n-trials 100 --device cuda:0 --out $O/RANK_ND_$sc.md > $O/RANK_ND_$sc.log 2>&1 || echo FAILED nd $sc
  for c in "2 32 Vanilla_r3ph" "2 10 Vanilla_r3ph" "3 18 Vanilla_r4ph" "3 10 Vanilla_r4ph"; do set -- $c
    python3 -m mapformer.rescore_hook --scale $sc -- mapformer.eval_nd --runs-dir $REPO/runs/rank_wrap/N$2 --configs "$1:$2:$3" --lengths 1024 \
      --n-trials 100 --device cuda:0 --out $O/RANK_WRAP_D$1_N$2_$sc.md > $O/RANK_WRAP_D$1_N$2_$sc.log 2>&1 || echo FAILED wrap $c $sc
  done
done
echo done
