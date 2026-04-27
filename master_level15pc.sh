#!/bin/bash
# master_level15pc.sh — autonomous pipeline for the Level15PC variant.
#
# Stages:
#   1. Wait for in-flight Level15PC trainings (clean s0, noise s0) to land
#   2. Run orchestrator_level15pc.py → completes 9 runs (3 configs × 3 seeds)
#   3. Run all evaluation scripts (long-seq, per-visit, calibration,
#      zero-shot, hippocampal_hidden_eval, clone_analysis)
#   4. Run orchestrator_finalize.sh → writes RESULTS_PAPER.md, commits + pushes
#   5. Append session findings to CLAUDE.md, commit + push

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs

is_running () {
    ps aux | grep -E "$1" | grep -v grep | grep -v master_level15pc >/dev/null
}

echo "[$(date)] stage 1 — wait for in-flight Level15PC s0 trainings..."
while is_running "train_variant.*Level15PC.*--seed 0"; do
    sleep 60
done
echo "[$(date)] in-flight Level15PC s0 jobs done."

# ----- Stage 2: orchestrator -----
echo "[$(date)] stage 2 — orchestrator_level15pc..."
nohup python3 -u -m mapformer.orchestrator_level15pc \
    > "$LOGS/orchestrator_level15pc.log" 2>&1 &
ORCH_PID=$!
wait $ORCH_PID
echo "[$(date)] orchestrator_level15pc done."

# ----- Stage 3: evaluations (parallel where possible) -----
ZS_VARIANTS="Vanilla VanillaEM Level15 Level15EM Level15PC PC LSTM MambaLike"

echo "[$(date)] stage 3a — long-seq evals..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.long_sequence_eval \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    > "$REPO/LONG_SEQ_clean.md" 2>"$LOGS/long_seq_clean.err" &
P_CLEAN=$!
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.long_sequence_eval \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    > "$REPO/LONG_SEQ_lm200.md" 2>"$LOGS/long_seq_lm200.err" &
P_LM=$!
wait $P_CLEAN $P_LM
echo "[$(date)] long-seq done."

echo "[$(date)] stage 3b — per-visit eval..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.per_visit_eval \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 --n-test-seeds 3 --T 512 \
    --output "$REPO/PER_VISIT_clean.md" 2>"$LOGS/per_visit_clean.err" &
PV_CLEAN=$!
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.per_visit_eval \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 --n-test-seeds 3 --T 512 \
    --output "$REPO/PER_VISIT_lm200.md" 2>"$LOGS/per_visit_lm200.err" &
PV_LM=$!
wait $PV_CLEAN $PV_LM
echo "[$(date)] per-visit done."

echo "[$(date)] stage 3c — zero-shot eval..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 --n-test-seeds 3 \
    --lengths 128 512 \
    --include-bias \
    --output "$REPO/ZERO_SHOT_TRANSFER_clean.md" 2>"$LOGS/zs_clean.err" &
ZS_C=$!
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.zero_shot_eval \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    --model-seeds 0 1 2 --n-test-seeds 3 \
    --lengths 128 512 \
    --include-bias --include-lm-sweep \
    --output "$REPO/ZERO_SHOT_TRANSFER_lm200.md" 2>"$LOGS/zs_lm200.err" &
ZS_LM=$!
wait $ZS_C $ZS_LM
echo "[$(date)] zero-shot done."

echo "[$(date)] stage 3d — hippocampal hidden eval..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.hippocampal_hidden_eval \
    --runs-dir mapformer/runs \
    --variants Vanilla Level15 PC Level15PC Grid_Free \
    --seed 0 --config clean \
    --T 256 --n-trials 30 \
    --output-md "$REPO/HIPPOCAMPAL_LEVEL15PC.md" \
    --output-figures "$REPO/paper_figures" 2>"$LOGS/hippo_l15pc.err" || true
echo "[$(date)] hippocampal hidden eval done."

echo "[$(date)] stage 3e — clone analysis..."
if [ -f "$REPO/clone_analysis.py" ]; then
    cd "$REPO"
    CUDA_VISIBLE_DEVICES=0 python3 -u clone_analysis.py \
        --variants Vanilla Level15 PC Level15PC \
        --config clean --seed 0 \
        > "$REPO/CLONE_ANALYSIS_LEVEL15PC.md" 2>"$LOGS/clone.err" || true
    cd /home/prashr
fi
echo "[$(date)] clone analysis done (or skipped)."

echo "[$(date)] stage 3f — calibration figures..."
cd /home/prashr
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.calibration_analysis \
    --runs-dir mapformer/runs --config clean \
    --variants $ZS_VARIANTS \
    --output "$REPO/paper_figures/calibration_clean.png" \
    > "$LOGS/calib_clean.log" 2>&1 || true
CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.calibration_analysis \
    --runs-dir mapformer/runs --config lm200 \
    --variants $ZS_VARIANTS \
    --output "$REPO/paper_figures/calibration_lm200.png" \
    > "$LOGS/calib_lm200.log" 2>&1 || true
echo "[$(date)] calibration done."

echo "[$(date)] stage 3g — per-visit figures..."
python3 -u -m mapformer.make_per_visit_figure \
    --clean "$REPO/PER_VISIT_clean.md" \
    --lm200 "$REPO/PER_VISIT_lm200.md" \
    --output-dir "$REPO/paper_figures" \
    > "$LOGS/per_visit_figs.log" 2>&1 || true
echo "[$(date)] per-visit figures done."

# ----- Stage 4: finalize (regenerate RESULTS_PAPER.md + commit + push) -----
echo "[$(date)] stage 4 — orchestrator_finalize.sh..."
cd "$REPO"
bash orchestrator_finalize.sh 2>&1 | tail -20

# ----- Stage 5: append CLAUDE.md log + commit + push -----
cd "$REPO"
cat >> CLAUDE.md << 'CLAUDE_EOF'

## Session 2026-04-26 — Level15PC + Grid + GridL15PC findings

### New model variants

- `Level15PC` (`model_level15_pc.py`): MapFormer-WM + Level 1.5 InEKF + PC aux
  loss on the standard backbone. Tests forward-model + inverse-model
  complementarity.
- `Grid` / `Grid_Free` (`model_grid.py`): multi-orientation path integrator
  with fixed (hex) or learnable orientations.
- `GridL15PC` / `GridL15PC_Free` (`model_grid_l15_pc.py`): Grid + Level 1.5
  + PC aux. Kitchen-sink test for hex emergence.

### Empirical findings

Hex emergence is NOT solved by architecture or correction stacking:
- `Grid_Free` clean s0: loss 0.021, hex orientations stayed but max
  per-module grid score 0.036 (0/22 modules > 0.3).
- `GridL15PC_Free` clean s0: loss 0.084, hidden-state max 0.052
  (worse than Grid_Free's 0.095). Adding L15+PC ACTIVELY REDUCED hex.
- §6.5/§6.10 falsification strengthens. Bottleneck is training
  objective, not correction toolkit.

`Level15PC` multi-seed sweep launched via `orchestrator_level15pc.py`,
results in RESULTS_PAPER.md, LONG_SEQ_*.md, PER_VISIT_*.md,
ZERO_SHOT_TRANSFER_*.md, HIPPOCAMPAL_LEVEL15PC.md, CLONE_ANALYSIS_LEVEL15PC.md.

### Honest checkpoint logging

- `runs/Grid_clean_200ep/seed0/Grid.pt`: stale, won't load with current
  code (state-dict key mismatch from cos_orient/sin_orient → orientation_angles).
- `runs/Level15EM_b5_lm200/seed2/`: diagnostic-only (alt safe-init
  experiment, kept untracked).

CLAUDE_EOF

git add -A
git commit -m "Level15PC + Grid + GridL15PC: new variants, autonomous eval pipeline

- model_level15_pc.py (new): Level 1.5 InEKF + PC aux loss on standard WM
- model_grid.py: MapFormerWM_Grid + MapFormerWM_Grid_Free
- model_grid_l15_pc.py (new): Grid + L15 + PC kitchen-sink
- train_variant.py: register all new variants + d_model arg
- train.py: --aux-coef support
- All eval scripts (long_seq, per_visit, zero_shot, calibration):
  added Level15PC import + VARIANT_CLS entry + default-variants
- orchestrator_level15pc.py + master_level15pc.sh: autonomous pipeline
- RESULTS_PAPER.md + LONG_SEQ_* + PER_VISIT_* + ZERO_SHOT_TRANSFER_* +
  HIPPOCAMPAL_LEVEL15PC.md + CLONE_ANALYSIS_LEVEL15PC.md regenerated
- CLAUDE.md: session findings (hex falsification, Level15PC sweep)
- paper_figures: calibration + per-visit refreshed
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3

echo "[$(date)] master_level15pc: ALL DONE."
