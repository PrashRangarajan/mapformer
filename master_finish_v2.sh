#!/bin/bash
# master_finish_v2.sh — post-CoPE-triage completion chain.
#
# Ordering rationale:
#   1. MambaLike (9 runs, ~1.5h): the key baseline for the Mamba
#      subsumption question. Highest priority.
#   2. Zero-shot transfer eval (~2-3h): run ZERO_SHOT_TRANSFER_{clean,lm200}.md
#      on every variant that has 3-seed coverage. Three axes (fresh obs_map,
#      biased actions, landmark-density sweep). Must run after MambaLike so
#      MambaLike appears in the eval tables.
#   3. Finalize #1: regenerate RESULTS_PAPER.md (incl. VanillaEM, Level15EM,
#      LSTM 9/9, CoPE 4/9, MambaLike 9/9 plus pre-existing). This commit also
#      catches the ZERO_SHOT_TRANSFER_*.md files written in step 2.
#   4. CoPE resume (5 runs, ~20h): backgrounded so it doesn't block
#      us inspecting results.
#   5. Finalize #2: once CoPE finishes, regenerate tables with complete
#      CoPE rows. Commit + push.
#
# Launch with:
#   nohup bash /home/prashr/mapformer/master_finish_v2.sh \
#       > /home/prashr/mapformer/master_finish_v2.log 2>&1 &

set -u
cd /home/prashr
REPO=/home/prashr/mapformer

echo "[$(date)] master_finish_v2: launching MambaLike orchestrator..."
nohup python3 -u -m mapformer.orchestrator_mambalike \
    > "$REPO/orchestrator_mambalike.log" 2>&1 &
ML_PID=$!
echo "  MambaLike PID: $ML_PID"
wait $ML_PID
echo "[$(date)] MambaLike done."

# ------------------------------------------------------------------
# Zero-shot transfer eval — three axes (fresh obs_map, biased actions,
# landmark-density). Runs across every variant with 3-seed coverage.
# Conservative args (3 test seeds, lengths [128, 512], 50 / 20 trials)
# keep wall time in the 2-3h range.
# ------------------------------------------------------------------
ZS_VARIANTS="Vanilla VanillaEM RoPE Level1 Level15 Level15EM PC LSTM MambaLike"

echo "[$(date)] Running zero-shot eval on config=clean (Axis 1 + Axis 2 only)..."
cd /home/prashr
python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 \
    --n-test-seeds 3 \
    --lengths 128 512 \
    --n-trials-128 50 \
    --n-trials-long 20 \
    --include-bias \
    --output "$REPO/ZERO_SHOT_TRANSFER_clean.md" \
    2>&1 | tail -5
echo "[$(date)] ZERO_SHOT_TRANSFER_clean.md written."

echo "[$(date)] Running zero-shot eval on config=lm200 (all 3 axes)..."
python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 \
    --n-test-seeds 3 \
    --lengths 128 512 \
    --n-trials-128 50 \
    --n-trials-long 20 \
    --include-bias \
    --include-lm-sweep \
    --output "$REPO/ZERO_SHOT_TRANSFER_lm200.md" \
    2>&1 | tail -5
echo "[$(date)] ZERO_SHOT_TRANSFER_lm200.md written."

echo "[$(date)] Running first finalize (all variants except CoPE noise/lm200)..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -15
echo "[$(date)] First finalize done. RESULTS_PAPER.md committed + pushed."

echo "[$(date)] Launching CoPE resume orchestrator (background, ~20h)..."
cd /home/prashr
nohup python3 -u -m mapformer.orchestrator_cope_resume \
    > "$REPO/orchestrator_cope_resume.log" 2>&1 &
CR_PID=$!
echo "  CoPE resume PID: $CR_PID"
wait $CR_PID
echo "[$(date)] CoPE resume done."

echo "[$(date)] Running second finalize (full CoPE rows now)..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -15

echo "[$(date)] master_finish_v2: all done."
