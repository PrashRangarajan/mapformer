# MapFormer: Self-Supervised Cognitive Maps with Lie Group Path Integration

Implementation of MapFormer ([Rambaud et al., 2025](https://arxiv.org/abs/2511.19279)),
faithfully reproducing the paper's results on 2D torus navigation, plus a
**parallel Invariant Extended Kalman Filter** extension for uncertainty-aware
path integration.

## What is MapFormer?

MapFormer is a Transformer that learns a **cognitive map** from an interleaved
sequence of (action, observation) tokens `s = (a₁, o₁, a₂, o₂, …)`. The model
must *learn by itself* to disentangle actions (which update position) from
observations (which update content). Key ideas:

- **Structure/content disentanglement**: actions drive a rotation in SO(2);
  observations leave position untouched.
- **Parallel path integration**: cumulative angles via `cumsum(ω·Δ)` + single
  `exp` — O(log T) on a parallel scan, length-generalises far beyond training.
- **Generalised RoPE**: rotations come from the learned action stream, not the
  token index.

## Models

| Component | Description |
|-----------|-------------|
| `MapFormer-WM` | Working Memory — generalised RoPE. Q/K rotated by learned action-dependent angles. |
| `MapFormer-EM` | Episodic Memory — absolute position embeddings from rotating learnable `p₀`. Attention combines content and structure via **Hadamard product** `softmax(A_X ⊙ A_P) V`. |
| `InEKF-Parallel` | **Our extension.** Invariant EKF on SO(2) with constant Q (Lie-group invariance), steady-state gain via closed-form scalar DARE, and FFT-based parallel affine scan. Preserves MapFormer's parallelism. |
| `PredictiveCoding` | **Our other extension.** Predictive-coding formulation: learned forward model `g(θ)→ô` predicts observation from position, prediction error drives Lie-algebra correction via learnable gate. More biologically plausible and handles aliased observations better than the inverse-model InEKF. |

## Project Structure

```
mapformer/
  environment.py         # Torus GridWorld, interleaved token sequences, revisit mask
  lie_groups.py          # SO(n) exp/log, block-diagonal rotations
  prefix_scan.py         # O(log T) parallel prefix product (for lie_groups utils)
  model.py               # MapFormer-WM and MapFormer-EM (paper-faithful)
  model_kalman.py        # First-pass InEKF (uncertainty-modulated attention — ablation)
  model_inekf_proper.py  # Proper sequential InEKF on SO(2) with wrapped innovations
  model_inekf_parallel.py# Steady-state InEKF with FFT-based parallel scan
  model_predictive_coding.py# Predictive-coding MapFormer (forward model + error)
  baselines.py           # Transformer+RoPE and LSTM baselines
  inekf.py               # Earlier InEKF scaffolding (pre-refactor; kept for reference)
  train.py               # Self-supervised training w/ revisit mask + optional action noise
  evaluate.py            # Length-generalisation + figures
  main.py                # Paper-faithful experiment pipeline
  main_kalman.py         # Train the first-pass InEKF
  main_inekf_proper.py   # Train the sequential proper InEKF
  main_inekf_parallel.py # Train the parallel InEKF
  main_predictive_coding.py# Train the predictive-coding MapFormer (with aux loss)
  main_vanilla_noise.py  # Train vanilla MapFormer with training-time action noise
  noise_test.py          # Discrete action-noise robustness test
  gaussian_noise_test.py # Gaussian Δ-noise robustness test (InEKF's home turf)
  diagnose.py            # Disentanglement + prediction distribution + ω analysis
  clone_analysis.py      # CSCG-style clone-structure analysis (per-obs-type)
  docs/                  # Original writeups
```

## Quick Start

```bash
pip install -r requirements.txt

# Reproduce the paper's Table 2 (2D navigation, 200K sequences).
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Scale up for cleaner near-perfect numbers (~1M sequences, ~30 min on a single GPU).
python3 -m mapformer.main --device cuda --epochs 50 --n-batches 156
```

Checkpoints are saved to `figures_<name>/*.pt`.

### Paper-matched defaults (from Rambaud et al. 2025, Appendix B)

- Grid: 64×64 **torus**, `p_empty=0.5`, 16 observation types + 1 blank
- Sequence: 128 (action, observation) steps → 256 interleaved tokens
- Model: d=128, 2 heads, head size h=64, **1 layer**
- Optimiser: AdamW, lr=3e-4, weight-decay=0.05, linear LR decay
- Batch size 128; 200K sequences total (16 epochs × 98 batches × 128)
- Loss: cross-entropy on observation tokens **at revisited locations only**

## Reproduced Results

| Model | Train acc (T=128) | 16× OOD (T=2048) |
|-------|-------------------|------------------|
| MapFormer-WM | **0.955** | 0.570 |
| MapFormer-EM | **0.999** | 0.544 |
| Paper (Table 2) | 0.99–1.0 | — |

## Critical Implementation Details (Things The Paper Glosses Over)

1. **Torus, not bounded grid.** Actions wrap via `(x+dx) % N`.
2. **Unified token stream.** Actions and observations share a single embedding
   table; the model must learn which tokens update position (‖Δ‖ large) vs
   content (‖Δ‖ ≈ 0). This disentanglement IS the cognitive-map learning.
3. **Revisit-only loss.** First-visit observations are informationally random
   and training on them collapses the model to "always predict blank".
   Loss must be masked to revisited locations only.
4. **ω initialisation is monotonic DECREASING in i.**
   The paper's eq. 17 reads `ω_i = ω_max · (1/Δ_max)^(-i/n_b)` which literally
   gives an *increasing* schedule growing to ~200 rad per token — causes
   catastrophic aliasing. The correct schedule (matching the paper's own
   `ω_min = ω_max/Δ_max` and RoPE analogy) is:
   ```python
   ω_i = ω_max · (1/Δ_max)^(i/(n_b-1))     # ω_max=2π, Δ_max=grid_size
   ```
5. **EM attention is Hadamard**, not additive. `softmax(A_X ⊙ A_P) V`.
6. **MapEM uses separate learnable `q_0^p` and `k_0^p`**, rotated by the
   path-integrated angles — not a linear projection of flattened rotation
   matrices.
7. **Low-rank Δ projection** `W_Δ = W_Δ^out · W_Δ^in` with bottleneck `r=2`
   (matching the 2D movement vector) and per-head, per-block outputs.

## InEKF Extension

We added an **Invariant Extended Kalman Filter** on SO(2) to stabilise path
integration under action/sensor noise. Three implementations:

1. `model_kalman.py` — first pass; modulates attention temperature by σ²
   (turned out to be redundant with attention's natural behaviour).
2. `model_inekf_proper.py` — proper Bayesian state correction with wrapped
   innovations. Sequential loop, ~2.5× slower than vanilla.
3. `model_inekf_parallel.py` — **steady-state parallel InEKF**. Key insights:
   - On SO(2), the Lie-Group EKF is provably equivalent to the EKF with
     `atan2`-wrapped innovation (Marković et al., 2017).
   - Scalar DARE has a closed-form fixed point → K* precomputed, constant
     across tokens.
   - Correction recurrence `d_t = (1-K*)·d_{t-1} + K*·ν_t` is a scalar
     affine recurrence, parallelisable via FFT convolution with kernel
     `K*·(1-K*)^k`.
   - Same speed as vanilla MapFormer (~10 s/epoch) despite adding state
     correction.

### Clone-Structure Analysis

Inspired by Dileep George et al.'s CSCG (Clone-Structured Cognitive Graph,
Nature Comms 2021), we ask: **do these models learn per-cell "clone"
representations when observations are aliased?** For each observation type,
we measure whether the model's internal state at the observation token
separates into distinct per-cell clusters.

On 300 trajectories from a fixed start, evaluating at observation positions:

| Model | θ̂ R² | hidden R² | θ̂ separation | hidden separation |
|-------|------|-----------|---------------|--------------------|
| Vanilla+noise | 0.184 | 0.366 | 0.573 | 0.1245 |
| **PC MapFormer** | 0.210 | 0.369 | **0.619** | 0.1472 |
| Parallel InEKF | **0.307** | 0.371 | 0.395 | **0.1706** |

- **R²**: linear decoding of (x, y) from feature; higher = position linearly
  recoverable
- **Separation**: `(mean_between_cell_dist − mean_within_cell_dist) /
  mean_between_cell_dist`, cosine distance; higher = more clone-like
  clustering

**PC has the cleanest clone structure in the Lie-algebra state (θ̂)** —
tightest per-cell clusters, most consistent with CSCG's discrete clone
interpretation. InEKF has a more continuous linear mapping but blurrier
clusters. See `clone_analysis.py`.

### What We Learned

**InEKF vs vanilla noise-augmentation** (Gaussian Δ perturbation, 10% noise):

| noise_std | Vanilla (noise aug) | Parallel InEKF | Best |
|-----------|---------------------|----------------|------|
| 0.00      | **0.97**            | 0.91           | Vanilla |
| 0.05      | **0.79**            | 0.67           | Vanilla |
| 0.10      | 0.59                | 0.51           | Vanilla |
| 0.50      | 0.44                | **0.46**       | InEKF (+1.4pp) |
| 1.00      | 0.47                | 0.44           | Tied |

**Honest takeaway:** On the standard MapFormer task, vanilla self-attention
already implements most of what the Kalman filter was supposed to add
(associative memory + soft retrieval = implicit Bayesian filter). Explicit
Kalman machinery earns its keep in regimes the paper doesn't test:

- **Very high noise** where path integration has fully diverged
- **Sparse explicit landmarks** (not tested — would require true unique IDs)
- **Sensor fusion with known noise spec** (Q, R given, not learned)
- **Downstream control loops** needing calibrated uncertainty
- **Very long horizons** where attention windows are infeasible

See `gaussian_noise_test.py` for the evaluation; `diagnose.py` for
disentanglement and per-token Δ analysis.

## Key Math

| Quantity | Formula |
|----------|---------|
| Path integration (algebra) | `θ_t = ω · cumsum(Δ)_t`  — parallel cumsum |
| Rotation (group) | `R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]`  — elementwise |
| RoPE (eq. 16) | `[x₁', x₂'] = [x₁ cos θ - x₂ sin θ,  x₁ sin θ + x₂ cos θ]` |
| WM attention | `softmax(Q̃ K̃ᵀ / √d) V`, with Q̃, K̃ rotated by θ_t |
| EM attention | `softmax(A_X ⊙ A_P) V`, with `A_P = Q_P K_Pᵀ / √d` |
| ω init (SO(2), Δ_max=grid, ω_max=2π) | `ω_i = ω_max · (1/Δ_max)^(i/(n_b-1))` |
| InEKF predict (SO(2)) | `θ̂ ← θ̂ + ω·Δ`,  `Σ ← Σ + Q` (state-independent) |
| InEKF innovation (SO(2)) | `ν = atan2(sin(z - θ̂), cos(z - θ̂))`  — geodesic |
| InEKF update | `θ̂ ← θ̂ + K·ν`,  `Σ ← (1-K)·Σ` |
| Steady-state DARE (scalar) | `P* = (-Q + √(Q² + 4QR))/2`,  `K* = (P*+Q)/(P*+Q+R)` |
| Parallel correction (FFT) | `d = ν * h`  where `h[k] = K*·(1-K*)^k` |

## Handoff Notes (for picking up this work)

### Current state (as of the last commit)

- **Paper reproduction: done.** MapFormer-WM and MapFormer-EM hit paper-level
  training accuracy (0.96 / 1.00) with all six paper-faithfulness fixes applied.
  Checkpoints live in `figures_v6/MapFormer_{WM,EM}.pt`.
- **Parallel InEKF: working.** `model_inekf_parallel.py` trains at vanilla
  speed (~10 s/epoch) with constant-gain DARE + FFT-scan. Checkpoint in
  `figures_inekf_parallel_v2/MapFormer_WM_ParallelInEKF.pt`.
- **Sequential InEKF: working but ~2.5× slower.** Two variants in
  `model_inekf_proper.py` (the current one wraps innovations; earlier git
  history had an unwrapped version that was faster to train but broke length
  generalisation).
- **Predictive-Coding variant: implemented, being tested.**
  `model_predictive_coding.py` — same parallel scan infrastructure as the
  InEKF, but uses a *forward model* (position → predicted obs embedding) and
  error-driven corrections instead of inverse-model Kalman. Handles aliased
  observations more gracefully in principle. Training uses an auxiliary
  prediction-error loss. Checkpoint in
  `figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt`.

### Key invariants (DON'T regress these)

1. **Environment returns `(tokens, obs_mask, revisit_mask)` per trajectory.**
   Loss is computed only on revisit positions; unmasked training collapses the
   model to "always predict blank."
2. **Grid is a torus.** `(x+dx) % N`, not `clip(x+dx, 0, N-1)`.
3. **Unified vocabulary.** Actions (0..3) and observations (4..4+K) share one
   embedding table; disentanglement must be learned.
4. **ω init is monotonically decreasing.** The paper's eq. 17 has a sign typo;
   verify via `model.path_integrator.omega` spans roughly `[2π/N, 2π]`.
5. **EM attention uses Hadamard product**, not additive.
6. **Measurement head in InEKF is content-only.** Feeding (cos θ̂, sin θ̂)
   creates a degenerate optimum (`z = θ̂` → trivial filter).

### Reproducing what we have

```bash
# Sanity-check that paper reproduction still works (~5 min on a modern GPU).
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train the parallel InEKF under action-noise augmentation (~10 min).
python3 -m mapformer.main_inekf_parallel --device cuda --epochs 50 --n-batches 156 \
    --p-action-noise 0.10

# Train the predictive-coding variant (~10 min).
python3 -m mapformer.main_predictive_coding --device cuda --epochs 50 --n-batches 156 \
    --p-action-noise 0.10 --aux-coef 0.1

# Compare robustness to Gaussian Δ noise at T=128 and T=512.
python3 -m mapformer.gaussian_noise_test \
    --checkpoints \
      figures_v6/MapFormer_WM.pt \
      figures_vanilla_noise/MapFormer_WM_noise.pt \
      figures_inekf_parallel_v2/MapFormer_WM_ParallelInEKF.pt \
    --device cuda --n-steps 128

# Diagnose a trained model: Δ per token, prediction distribution, ω, revisit
# accuracy, action selectivity (N+S should ≈ 0).
python3 -m mapformer.diagnose --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

### Open questions / next steps

1. **Level 2 InEKF: time-varying K_t via heteroscedastic noise.** Still
   parallelisable via associative scan of Möbius 2×2 matrices. Should help if
   we add true landmarks (see §below). See my longer explanation in the chat
   history.
2. **Add true landmarks to the environment.** A small fraction (~5%) of cells
   emit unique tokens found nowhere else on the map. This is the regime the
   InEKF framework was actually designed for; current experiments test noise
   without landmarks, which is why attention's implicit retrieval dominates.
3. **Calibration metrics.** Currently we only measure accuracy. Adding NLL or
   ECE would show whether InEKF-tracked σ² corresponds to actual
   uncertainty — a useful property for downstream planning.
4. **Scaling.** Paper itself notes "we did not scale to larger model/data."
   Our 1-layer, 2-head, d=128 model is tiny. Try 4 layers, 4 heads, d=256 to
   see if EM reaches 100% OOD at 16× length.
5. **Implement MapEM-s and MapEM-o ablations.** The current `MapFormerEM`
   implements only MapEM-os (observation AND structure). The paper also tests
   MapEM-s (structure only) and MapEM-o (content only, as a control).
6. **Fix broken `figures_inekf_proper` checkpoint.** We trained an unwrapped
   InEKF there, then edited `model_inekf_proper.py` to add wrapping, which
   makes the old checkpoint mismatched with current code. Either delete or
   re-train.

### Pitfalls we hit (so you don't repeat them)

- **ω exploded to ~200.** Faithfully reproducing the paper's eq. 17 typo —
  check `model.py::PathIntegrator.__init__` uses the decreasing schedule.
- **Model collapsed to predicting blank (94% of tokens).** Root cause: loss on
  first-visit observations = training on random noise. Always use
  `revisit_mask`.
- **"Uncertainty-modulated attention" did nothing.** First-pass InEKF in
  `model_kalman.py` — redundant with what softmax attention already does.
  Don't revive this approach unless you have a specific reason.
- **Position-conditioned measurement head = degenerate optimum.** The head
  learns `z ≈ θ̂`, innovation → 0, filter becomes identity. The
  `model_inekf_parallel.py` header comment warns about this.
- **Checkpoint filename collisions.** `figures_inekf_proper/` and
  `figures_inekf_topology_fix/` both saved as `MapFormer_WM_ProperInEKF.pt`.
  Rename or delete one.

### File ownership map

| File | Author | Status |
|------|--------|--------|
| `environment.py`, `model.py`, `train.py`, `evaluate.py`, `main.py`, `baselines.py`, `lie_groups.py`, `prefix_scan.py` | Paper reproduction | Stable |
| `inekf.py` | Pre-refactor InEKF scaffolding | Kept for reference, unused by current pipeline |
| `model_kalman.py`, `main_kalman.py` | Ablation (uncertainty-modulated attention) | Kept for comparison |
| `model_inekf_proper.py`, `main_inekf_proper.py` | Sequential proper InEKF | Working |
| `model_inekf_parallel.py`, `main_inekf_parallel.py` | **Parallel InEKF (main contribution)** | Working |
| `model_predictive_coding.py`, `main_predictive_coding.py` | **Predictive-coding MapFormer** (forward model + error-driven correction) | Working |
| `main_vanilla_noise.py` | Vanilla + noise-aug baseline | For fair comparison |
| `noise_test.py`, `gaussian_noise_test.py` | Robustness evaluation | Stable |
| `diagnose.py` | Introspection (Δ disentanglement, ω, etc.) | Stable |

## References

- Rambaud, Mascarenhas, Lakretz (2025). *MapFormer: Self-Supervised Learning of
  Cognitive Maps with Input-Dependent Positional Embeddings.*
  [arXiv:2511.19279](https://arxiv.org/abs/2511.19279)
- Barrau, Bonnabel (2017). *The Invariant Extended Kalman Filter as a Stable
  Observer.* IEEE TAC, 62(4). [arXiv:1410.1465](https://arxiv.org/abs/1410.1465)
- Marković, Ćesić, Petrović (2017). *On wrapping the Kalman filter and
  estimating with the SO(2) group.*
  [arXiv:1708.05551](https://arxiv.org/abs/1708.05551)
- Särkkä, García-Fernández (2021). *Temporal Parallelization of Bayesian
  Filters and Smoothers.* IEEE TAC, 66(1).
  [arXiv:1905.13002](https://arxiv.org/abs/1905.13002)
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position
  Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
