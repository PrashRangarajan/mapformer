#!/bin/bash
# Runs after the main orchestrator finishes. Chains:
#   1. orchestrator_finalize.sh (aggregated paper tables)
#   2. long_sequence_eval at T up to 2048, pushing boundaries
#   3. calibration analysis figure for each main config
#   4. paper figure generation
#   5. multilayer experiments (will run on freed GPUs)
# Each step commits+pushes on completion.

set -e
cd /home/prashr

REPO=/home/prashr/mapformer

echo "[$(date)] Waiting for main orchestrator to finish..."
# Wait for RESULTS_PAPER.md to appear (finalize script writes it)
while ! [[ -s "$REPO/RESULTS_PAPER.md" ]]; do
  # Check if orchestrator is still alive
  if ! pgrep -f "mapformer.orchestrator " > /dev/null; then
    if ! [[ -s "$REPO/RESULTS_PAPER.md" ]]; then
      echo "[$(date)] Main orchestrator died without producing RESULTS_PAPER.md"
      echo "[$(date)] Running orchestrator_finalize.sh manually..."
      cd "$REPO" && bash orchestrator_finalize.sh || true
      cd /home/prashr
    fi
    break
  fi
  sleep 60
done
echo "[$(date)] Main orchestrator done."

# 1. Long-sequence evaluation
echo "[$(date)] Running long-sequence eval..."
python3 -u -m mapformer.long_sequence_eval \
    --runs-dir mapformer/runs --config clean \
    --lengths 128 256 512 1024 2048 \
    --variants Vanilla RoPE Level1 Level15 PC \
    --output "$REPO/LONG_SEQ_clean.md" > "$REPO/LONG_SEQ_clean.md" 2>&1 || true

python3 -u -m mapformer.long_sequence_eval \
    --runs-dir mapformer/runs --config lm200 \
    --lengths 128 256 512 1024 2048 \
    --variants Vanilla RoPE Level1 Level15 PC \
    --output "$REPO/LONG_SEQ_lm200.md" > "$REPO/LONG_SEQ_lm200.md" 2>&1 || true

# 2. Calibration
echo "[$(date)] Running calibration analysis..."
cd /home/prashr
python3 -u -m mapformer.calibration_analysis \
    --runs-dir mapformer/runs --config clean \
    --output "$REPO/paper_figures/calibration_clean.png" || true
python3 -u -m mapformer.calibration_analysis \
    --runs-dir mapformer/runs --config lm200 \
    --output "$REPO/paper_figures/calibration_lm200.png" || true

# 3. Paper figures
echo "[$(date)] Generating paper figures..."
python3 -u -m mapformer.make_paper_figures \
    --runs-dir mapformer/runs --output-dir "$REPO/paper_figures" || true

# Commit & push everything
cd "$REPO"
git add LONG_SEQ_*.md paper_figures/ paper/
git commit -m "Post-orchestrator: long-seq evals, calibration, figures, paper drafts" || true
git push origin main 2>&1 | tail -3

# 4. Multi-layer orchestrator — queue after first set finishes
echo "[$(date)] Launching multi-layer experiments..."
cd /home/prashr
nohup python3 -u -m mapformer.orchestrator_multilayer > "$REPO/orchestrator_multilayer.log" 2>&1 &
ML_PID=$!
echo "  Multi-layer orchestrator PID: $ML_PID"

# 5. Wait for multi-layer to finish, then run extra baselines
wait $ML_PID
echo "[$(date)] Multi-layer done. Launching extra baselines..."
nohup python3 -u -m mapformer.orchestrator_baselines > "$REPO/orchestrator_baselines.log" 2>&1 &
BL_PID=$!
echo "  Extra-baselines PID: $BL_PID"
wait $BL_PID
echo "[$(date)] Extra baselines done."

# Final re-evaluation with all variants and commit
cd /home/prashr/mapformer
bash orchestrator_finalize.sh 2>&1 | tail -5
echo "[$(date)] Followup pipeline fully done."
