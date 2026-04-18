# CLAUDE.md — Project Memory for MapFormer

**Purpose of this file:** concise context for Claude when resuming work on this
project in a fresh session. Read the README for the full picture; this file
focuses on state + lessons learned + what to do next.

## Project in one sentence

Faithful reproduction of Rambaud et al. (2025) *MapFormer* (arXiv:2511.19279)
plus three experimental extensions that add explicit state-correction
mechanisms to the path-integration circuit: a **parallel Invariant EKF**, a
**sequential InEKF** (for reference), and a **predictive-coding** variant.

## Current state (what's implemented and working)

1. **Paper reproduction** — `model.py`, `environment.py`, `main.py`.
   MapFormer-WM / EM reach paper-level accuracy (0.955 / 0.999) on 200K
   sequences at the paper's exact hyperparameters (Appendix B). Checkpoints
   in `figures_v6/`.
2. **Parallel InEKF** — `model_inekf_parallel.py`, `main_inekf_parallel.py`.
   Steady-state gain + FFT scan. Same speed as vanilla. Checkpoint in
   `figures_inekf_parallel_v2/`.
3. **Sequential InEKF (wrapped)** — `model_inekf_proper.py`,
   `main_inekf_proper.py`. ~2.5× slower but same final accuracy as parallel.
   Checkpoint in `figures_inekf_topology_fix/`.
4. **Predictive-Coding MapFormer** — `model_predictive_coding.py`,
   `main_predictive_coding.py`. Forward model + error-driven corrections.
   Checkpoint in `figures_predictive_coding/` after training completes.
5. **Evaluation tools** — `noise_test.py`, `gaussian_noise_test.py`,
   `diagnose.py`. Handle all variants above.

## Things that are true (verified) and must be preserved

**Paper-faithfulness invariants** (breaking any of these regresses to
broken states we debugged through):

- `environment.py`: torus grid (`(x+dx) % N`), interleaved token stream
  `[a1, o1, a2, o2, …]`, `revisit_mask` returned per trajectory
- `train.py`: loss masked to **revisited** observation positions only
- `model.py::PathIntegrator`: ω initialized monotonically decreasing in `i`
  (paper eq. 17 has a sign typo; the correct formula is
  `ω_i = ω_max · (1/Δ_max)^(i/(n_b-1))`)
- `model.py::MapFormerEM`: attention is Hadamard product `A_X ⊙ A_P`, not
  additive; uses separate learnable `q_0^p` and `k_0^p`
- `model.py::ActionToLieAlgebra`: low-rank factorization `W_out · W_in`
  with bottleneck `r=2`
- `model.py`: path integration via cumsum of angles, not prefix product of
  rotation matrices
- Default hyperparameters: 1 layer, 2 heads, h=64, d_model=128, lr=3e-4,
  AdamW, wd=0.05, linear LR decay, batch 128, grid 64, T=128 steps,
  K=16 obs types, p_empty=0.5, 200K sequences

## Architectural choices that matter

- **Feed `content_emb` only to the InEKF measurement head.** Adding
  position features `(cos θ, sin θ)` creates a degenerate optimum
  `z ≈ θ` → zero innovation → filter does nothing. We learned this
  the hard way.
- **Wrap innovations modulo 2π** via `atan2(sin(z - θ̂), cos(z - θ̂))`.
  Without this, length generalization breaks (θ̂ grows unboundedly, bounded
  z can't express the large "error" geometrically).
- **Steady-state Kalman gain from closed-form scalar DARE.** Enables
  FFT-conv based parallel affine scan, preserving MapFormer's O(log T)
  property.
- **Markovic et al. (2017) proves:** on SO(2), wrapped-innovation EKF equals
  Lie-Group EKF. So the simple wrapping *is* the correct Lie-group filter.
- **Predictive coding uses a forward model** `g(cos θ, sin θ) → ô` and
  computes error in *embedding space*, masked at observation positions only
  (action positions have unpredictable content conditioned on θ alone).
  Includes an auxiliary loss coefficient to force the forward model to
  actually model observations.

## Main empirical finding

On the paper's aliased-observation task, **vanilla attention + noise
augmentation beats all Kalman-style variants on raw next-token prediction**.
Reasons, in order of importance:

1. Attention already implements soft associative retrieval — implicit
   Bayesian pattern completion — which is what the Kalman update was meant
   to add.
2. Aliased observations (16 obs types / 4096 cells = ~128 cells per type)
   mean Kalman measurements can't produce sharp corrections. The Gaussian
   assumption of Kalman is violated; the true posterior is multimodal.
3. Innovation wrapping, required for length generalization, slows training
   by bounding per-step corrections.

**Where the Kalman / PC framework should win:**

- Tasks with **true landmarks** (5% of cells emit unique IDs found nowhere
  else). Not yet tested. Predicted to be where Kalman dominates.
- Very long sequences (T >> 2048) where attention becomes infeasible.
- Scenarios needing calibrated uncertainty (InEKF tracks σ², attention
  doesn't).
- External sensor fusion with known Q, R matrices.

## Things that didn't work / why

- **Uncertainty-modulated attention** (`model_kalman.py`, the first-pass
  InEKF): redundant with softmax attention's natural behavior. Kept for
  reference comparison.
- **Unwrapped InEKF innovations**: trained faster (0.66 vs 0.88 final loss)
  but broke at T=512 OOD — the measurement head extrapolated badly outside
  the short-sequence θ range it was trained on.
- **Adding position features to InEKF head**: degenerate optimum — the
  filter turns into the identity function.
- **Multi-layer MapFormer with shared θ**: not tested empirically. The
  paper only runs 1-layer MapFormer. If you need more layers for position
  correction, Option 1 (per-layer θ correction) is the natural extension
  but requires validation that multi-layer MapFormer trains stably first.

## Open questions / natural next experiments

1. **Evaluate the predictive-coding variant** against InEKF + vanilla on
   `gaussian_noise_test.py` at T=128 and T=512, noise_std 0.00 – 1.00.
   (This is running as of the last commit — check
   `figures_predictive_coding/`.)
2. **Add true landmarks** to `environment.py`: reserve ~5% of cells to
   emit unique high-info tokens (beyond the 16 standard types). Retrain +
   compare. This is the Kalman framework's home turf.
3. **Level 2 InEKF** (time-varying R_t from heteroscedastic head,
   parallelisable via Möbius-matrix associative scan). Theoretical sketch
   lives in the chat transcript; not yet implemented.
4. **Calibration metrics.** NLL or ECE would show whether InEKF's tracked
   σ² is a useful confidence estimate even when point accuracy doesn't
   improve.
5. **Multi-layer MapFormer ablation.** Paper only runs 1 layer. Test 2/3/4
   layers at vanilla + InEKF-augmented configurations to see whether depth
   helps at all in this architecture.
6. **Scaling.** Paper acknowledges they didn't scale model/data. 4 layers,
   4 heads, d=256, 10M sequences would be a natural next step.

## Quick reproducibility commands

```bash
# Re-verify paper reproduction:
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train each variant under 10% action-noise augmentation:
python3 -m mapformer.main_vanilla_noise --device cuda --epochs 50 --n-batches 156
python3 -m mapformer.main_inekf_parallel --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10
python3 -m mapformer.main_predictive_coding --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10 --aux-coef 0.1

# Head-to-head evaluation under Gaussian Δ noise:
python3 -m mapformer.gaussian_noise_test \
    --checkpoints \
      figures_v6/MapFormer_WM.pt \
      figures_vanilla_noise/MapFormer_WM_noise.pt \
      figures_inekf_parallel_v2/MapFormer_WM_ParallelInEKF.pt \
      figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --n-steps 128 --n-trials 200
# Then with --n-steps 512 for OOD length.

# Diagnostics on any trained model:
python3 -m mapformer.diagnose --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

## Filesystem map

- `figures_v6/` — paper-faithful MapFormer-WM and EM (reference)
- `figures_vanilla_noise/` — vanilla + noise-aug baseline
- `figures_inekf_parallel_v2/` — parallel InEKF main result
- `figures_inekf_topology_fix/` — sequential wrapped InEKF (same model class)
- `figures_inekf_proper/` — **stale**: sequential unwrapped InEKF
  (model code has since been updated; loading this checkpoint with current
  code is incorrect)
- `figures_kalman/` — first-pass fake InEKF (kept for comparison)
- `figures_predictive_coding/` — PC MapFormer (being populated)
- `figures_2M/`, `figures_constlr/`, `figures_v3/`, `figures_v4*/` — older
  runs from earlier debug sessions; can safely be deleted

## Authoring style / preferences

- No emojis in source code or commit messages
- No `Co-Authored-By` lines; single-author commits
- README is the primary documentation, this file is a memory-aid for Claude
- Honest reporting: if an experiment didn't work, write that down with the
  reason, don't bury it
