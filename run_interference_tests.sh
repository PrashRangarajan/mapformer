#!/bin/bash
# run_interference_tests.sh — autonomous runner for the three interference tests.
#
# Test 1: R_t distribution by token type (Level15 vs Level15PC, lm200)
#   ~3 min, GPU 0
#
# Test 2: aux_coef sweep on lm200 (5 trainings + eval)
#   ~30 min on 2 GPUs
#
# Test 3: Clone-separation transfer (in-dist vs OOD env)
#   ~10 min, GPU 0
#
# Strategy:
#   - Tests 1, 3 are eval-only (cheap): run them first while GPUs are free.
#   - Test 2 is training-heavy: run after.
#   - Commit + push at the end.
#
# Launch:
#   nohup bash /home/prashr/mapformer/run_interference_tests.sh \
#     > /home/prashr/mapformer/logs/interference_tests.log 2>&1 &

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "[$(date)] Starting interference tests..."

# Test 1: R_t distribution (eval-only, ~3 min)
echo "[$(date)] Test 1: R_t distribution by token type..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.r_t_distribution_test \
    --runs-dir mapformer/runs --config lm200 --seed 0 \
    --T 512 --n-trials 30 \
    --output "$REPO/R_T_DISTRIBUTION.md" 2>"$LOGS/test1_r_t_dist.err"
echo "[$(date)] Test 1 done."

# Test 3: clone-separation transfer (eval-only, ~10 min) — start in parallel with Test 2
echo "[$(date)] Test 3: clone-separation transfer (background)..."
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.clone_transfer_test \
    --runs-dir mapformer/runs --config lm200 --seed 0 \
    --variants Vanilla Level1 Level15 PC Level15PC \
    --T 128 --n-trials 200 \
    --output "$REPO/CLONE_TRANSFER_TEST.md" \
    > "$LOGS/test3_clone_transfer.log" 2>&1 &
TEST3_PID=$!

# Test 2: aux_coef sweep (training-heavy, both GPUs)
# This script internally manages 2-GPU parallelism.
echo "[$(date)] Test 2: aux_coef sweep launching after Test 3 finishes..."
wait $TEST3_PID
echo "[$(date)] Test 3 done. Now Test 2 (sweep)..."
python3 -u -m mapformer.aux_coef_sweep \
    --coefs 0.0 0.01 0.03 0.1 0.3 \
    --seed 0 --epochs 50 --n-batches 156 \
    --T-eval 512 --n-eval-trials 100 \
    --output "$REPO/AUX_COEF_SWEEP.md" 2>"$LOGS/test2_aux_sweep.err"
echo "[$(date)] Test 2 done."

# Commit + push
cd "$REPO"
git add R_T_DISTRIBUTION.md CLONE_TRANSFER_TEST.md AUX_COEF_SWEEP.md \
        r_t_distribution_test.py clone_transfer_test.py aux_coef_sweep.py \
        run_interference_tests.sh
# Also stage any new training-run dirs from Test 2
git add runs/Level15PC_lm200_aux*/seed0/Level15PC.pt 2>/dev/null || true
git add logs/Level15PC_lm200_aux*_s0.log 2>/dev/null || true

git commit -m "Three interference tests for Level15PC: R_t distribution, aux_coef sweep, clone transfer

Test 1 (R_T_DISTRIBUTION.md): mean log_R_t broken out by token type
  (action / blank / aliased / landmark) for Level15 vs Level15PC on lm200.
  Tests whether PC's aux loss flattens the per-token-type R gating.

Test 2 (AUX_COEF_SWEEP.md): trains Level15PC at aux_coef in
  {0, 0.01, 0.03, 0.1, 0.3} on lm200 single-seed. Reports OOD T=512 acc/NLL.
  If interference is the mechanism: monotonic dose-response curve.

Test 3 (CLONE_TRANSFER_TEST.md): clone-separation score on training env
  vs OOD env (fresh obs_map seed=10000). If PC's clone-structure win
  collapses OOD, it was memorisation; if it persists, it's a real
  transferable feature.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3

echo "[$(date)] Interference tests pipeline complete."
