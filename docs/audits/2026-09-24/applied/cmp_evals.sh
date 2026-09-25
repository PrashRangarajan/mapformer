#!/usr/bin/env bash
# usage: cmp_evals.sh <dir-A> <dir-B>   (outputs of run_evals.sh, or the repo for committed files)
for f in RANK_MATCHED_e900.json RANK_MATCHED_e900.md RANK_MATCHED_e900_STRATA.json RANK_MATCHED_e900_GEOMETRY.md \
         RANK_PERHEAD_PILOT_GEOMETRY.md RANK_PROJ_TRAIN.json RANK_PROJ_TRAIN.md RANK_PROJ_TRAIN_STRATA.json \
         RANK_MATCHED_e900_ANALYSIS.txt RANK_MATCHED_e900c_ANALYSIS.txt RANK_PROJ_ANALYSIS.txt RANK_PERHEAD_PILOT_ANALYSIS.txt; do
  if cmp -s "$1/$f" "$2/$f"; then echo "IDENTICAL $f"; else echo "DIFFERS   $f"; diff "$1/$f" "$2/$f" | head -8; fi
done
for t in strata_e900 strata_proj arm_e900; do echo "time $t: $(cat $1/time_$t.txt 2>/dev/null) -> $(cat $2/time_$t.txt 2>/dev/null)"; done
