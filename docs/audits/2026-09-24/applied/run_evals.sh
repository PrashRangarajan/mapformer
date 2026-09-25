#!/usr/bin/env bash
# usage: run_evals.sh <code-parent-dir> <out-dir>
# runs the rank evaluators + analyses with the `mapformer` package found under <code-parent-dir>
set -u
CP="$1"; O="$2"; mkdir -p "$O"; M=/home/prashr/mapformer
cd "$CP"
S="0 1 2 3 4 5 6 7"
(
python3 -u -m mapformer.eval_noise_refine --runs-dir $M/runs/rank_matched_e900 --variants Vanilla Vanilla_r4 --noises 0.0 \
  --seeds $S --lengths 512 1024 2048 --n-trials 100 --device cuda:0 \
  --title "Rank at matched length: trained AND tested at T=1024 (epochs=900)" --out $O/RANK_MATCHED_e900.md > $O/log_enr_e900.txt 2>&1
/usr/bin/time -f "%e s" -o $O/time_strata_e900.txt python3 -u -m mapformer.eval_rank_strata --runs-dir $M/runs/rank_matched_e900 --variants Vanilla Vanilla_r4 --seeds $S \
  --lengths 1024 2048 --n-trials 100 --device cuda:0 --check-json $M/RANK_MATCHED_e900.json --out $O/RANK_MATCHED_e900_STRATA.json > $O/log_strata_e900.txt 2>&1
python3 -u -m mapformer.probe_action_geometry --runs-dir $M/runs/rank_matched_e900/p0 --arms Vanilla Vanilla_r4 --seeds $S --device cuda:0 \
  --out $O/RANK_MATCHED_e900_GEOMETRY.md > $O/log_geo_e900.txt 2>&1
python3 -u -m mapformer.probe_action_geometry --runs-dir $M/runs/rank_perhead_pilot/p0 --arms Vanilla_r2ph --seeds 0 1 --device cuda:0 \
  --out $O/RANK_PERHEAD_PILOT_GEOMETRY.md > $O/log_geo_ph.txt 2>&1
echo done > $O/.gpu0_done
) &
(
python3 -u -m mapformer.eval_noise_refine --runs-dir $M/runs/rank_proj_train --variants Vanilla --noises 0.0 \
  --seeds $S --lengths 512 1024 2048 --n-trials 100 --device cuda:1 \
  --title "Projected r=2, trained 900 epochs (continuation recipe)" --out $O/RANK_PROJ_TRAIN.md > $O/log_enr_proj.txt 2>&1
/usr/bin/time -f "%e s" -o $O/time_strata_proj.txt python3 -u -m mapformer.eval_rank_strata --runs-dir $M/runs/rank_proj_train --variants Vanilla --seeds $S --lengths 1024 \
  --n-trials 100 --device cuda:1 --check-json $M/RANK_PROJ_TRAIN.json --out $O/RANK_PROJ_TRAIN_STRATA.json > $O/log_strata_proj.txt 2>&1
/usr/bin/time -f "%e s" -o $O/time_arm_e900.txt python3 -u -m mapformer.analyze_rank_matched --tag _e900 > $O/RANK_MATCHED_e900_ANALYSIS.txt 2>&1
python3 -u -m mapformer.analyze_rank_matched --tag _e900c > $O/RANK_MATCHED_e900c_ANALYSIS.txt 2>&1
python3 -u -m mapformer.analyze_rank_proj > $O/RANK_PROJ_ANALYSIS.txt 2>&1
python3 -u -m mapformer.analyze_rank_perhead > $O/RANK_PERHEAD_PILOT_ANALYSIS.txt 2>&1
echo done > $O/.gpu1_done
) &
wait
echo ALLDONE > $O/.all_done
