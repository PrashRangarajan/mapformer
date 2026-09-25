#!/usr/bin/env bash
# usage: run_evals_extra.sh <code-parent-dir> <out-dir> : eval_noise_refine on two older committed
# batches with other model classes (Looped/LoopedRefine/Level15 under action noise; the L15 ablation arms)
set -u
CP="$1"; O="$2"; mkdir -p "$O"; M=/home/prashr/mapformer; cd "$CP"
( python3 -u -m mapformer.eval_noise_refine --runs-dir $M/runs/noise_refine --variants Vanilla Looped LoopedRefine Level15 \
    --noises 0.0 0.10 0.25 --seeds 0 1 2 --lengths 128 512 --device cuda:0 --out $O/NOISE_REFINE.md > $O/log_noise_refine.txt 2>&1 ) &
( python3 -u -m mapformer.eval_noise_refine --runs-dir $M/runs/l15_ablation --variants Vanilla Level15 L15_NoMeas L15_NoCorr L15_ConstR L15_DARE \
    --noises 0.0 --seeds 0 1 2 3 4 --lengths 128 512 1024 --n-trials 100 --device cuda:1 --out $O/L15_ABLATION.md > $O/log_l15.txt 2>&1 ) &
wait; echo done > $O/.extra_done
