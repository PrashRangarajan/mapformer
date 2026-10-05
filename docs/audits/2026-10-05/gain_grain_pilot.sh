#!/usr/bin/env bash
# GAIN_GRAIN pilot (seeds outside the batch's 26-45), launched after the prereg commit:
# (1) reproduction -- MapPoPE-Pair s10 through train_gain_grain (which also imports model_em_pope), full 300 epochs,
#     against the stored runs/mappope_pair/p0/MapPoPE-Pair_s10 (bitwise per-epoch losses);
# (2) sanity + timing at the batch's concurrency (8 jobs, 4 per GPU), full recipe: GainScalar, GainMod4, VanillaEM,
#     VanillaEM_NonNeg at s100; GainScalar, GainMod4, VanillaEM_NonNeg at s101;
# (3) the batch's eval path (eval_gain_grain -> eval_noise_refine, ckpt_guard layout) on the pilot runs.
# Flags = run_gain_grain.sh's.
set -uo pipefail
REPO=/home/prashr/mapformer; P="$REPO/runs/gain_grain_pilot"; LOG="$P/pilot.log"
mkdir -p "$P"; source "$REPO/lib_driver.sh"; DRV_SPACING=3
cd "$REPO/.."
go(){ local V=$1 S=$2 G=$3 TAG=$4 OUT="$P/$4/${1}_s${2}"; mkdir -p "$OUT"
  OMP_NUM_THREADS=2 drv_launch "$P/$TAG/${V}_s${S}.log" python3 -u -m mapformer.train_gain_grain --variant "$V" --seed "$S" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --lr 1e-3 --device "cuda:$G" --output-dir "$OUT"; }
echo "start $(date)" >> "$LOG"
go MapPoPE-Pair 10 0 repro
go GainScalar 100 0 p0; go GainMod4 100 0 p0; go VanillaEM 100 0 p0
go VanillaEM_NonNeg 100 1 p0; go GainScalar 101 1 p0; go GainMod4 101 1 p0; go VanillaEM_NonNeg 101 1 p0
drv_wait_dir "$P/"
echo "training done $(date)" >> "$LOG"
python3 -u -m mapformer.eval_gain_grain mapformer.eval_noise_refine --runs-dir "$P" \
  --variants GainScalar GainMod4 VanillaEM VanillaEM_NonNeg --noises 0.0 --seeds 100 101 --lengths 128 --n-trials 100 \
  --device cuda:0 --out "$P/PILOT_EVAL.md" --title "GAIN_GRAIN pilot (seeds 100-101)" >> "$LOG" 2>&1
touch "$P/.pilot_done"; echo "done $(date)" >> "$LOG"
