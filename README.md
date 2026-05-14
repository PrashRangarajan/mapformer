# MapFormer + Parallel Invariant EKF: Cognitive Maps with Calibrated State Correction

This repository implements [MapFormer (Rambaud et al. 2025)](https://arxiv.org/abs/2511.19279)
with high paper-faithfulness, then extends it with a family of **parallel
Invariant Extended Kalman Filter** (InEKF) corrections and a
**Gaussian Sum Filter** (multi-modal Bayes) variant. The work tests
whether MapFormer's cognitive-map inductive bias is *necessary* for path-
integration tasks, identifies architectural choices that close the gap to
the strongest existing baseline (TEMFaithful), and probes for hippocampal-
like representations.

The detailed experiment-by-experiment record, ablation details, and
session-by-session history are in
[DETAILED_RESULTS.md](DETAILED_RESULTS.md).
The session-2026-05-10 walkthrough is in
[RESULTS_SUMMARY_2026-05-10.md](RESULTS_SUMMARY_2026-05-10.md)
(also as PDF).

## Headline results

### Cognitive-map necessity (six independent cognitive demands)

Standard transformer (RoPE) collapses across every cognitive demand we
tested; MapFormer family with correction wins all of them. Numbers are
held-out accuracy at the hardest condition for each demand:

| Cognitive demand                          | RoPE   | MapFormer best         | Gap     |
| ----------------------------------------- | ------ | ---------------------- | ------- |
| Long-T extrapolation (T=2048, lm200)      | 0.465  | 0.835 (GSF_NoDrop)     | +37pp   |
| Sparse landmarks (lm10, T=512 OOD)        | 0.519  | 0.997 (GSF_NoDrop)     | +48pp   |
| Multi-env held-out (50 train/50 test)     | 0.506  | 0.988 (Level15)        | +48pp   |
| OOD-s (128×128, p_empty=0.8, T=512)       | 0.741  | 0.984 (GSF_NoDrop)     | +24pp   |
| Multi-size held-out (32, 64, 128 mixed)   | 0.490  | 0.975 (K=16 base)      | +49pp   |
| Cross-topology (torus/open/walls mixed)   | 0.519  | 0.956 (K=16 base)      | +44pp   |

In every regime, standard transformer is at or below chance for the task;
MapFormer family near ceiling. The cognitive-map inductive bias is *necessary*,
not incremental.

### Match or beat the strongest cognitive-map baseline (TEMFaithful)

After a critical bug fix (predict-then-update order), TEMFaithful jumps
from 0.42 to 0.969 on lm200 OOD T=512 and becomes the strongest existing
comparison. Two independent architectural changes inside MapFormer match
or beat TEMFaithful at the regimes where it was designed to excel:

| Variant                  | lm200 T=512  | LongT T=2048 | Sparse lm10 T=512 |
| ------------------------ | ------------ | ------------ | ----------------- |
| Level15 (baseline)       | 0.819 ± 0.025| 0.654 ± 0.031| 0.987             |
| Level15NoDrop            | 0.948 ± 0.025| —            | —                 |
| Level15GSF               | 0.956 ± 0.042| 0.835 ± 0.060| 0.997             |
| **Level15GSF_NoDrop**    | **0.961 ± 0.038** | **0.835 ± 0.060** | **0.997** |
| TEMFaithful (post-fix)   | 0.969 ± 0.010| 0.734 ± 0.024| 0.977             |

- **Level15GSF_NoDrop ≈ TEMFaithful at lm200** (within seed std)
- **Level15GSF_NoDrop beats TEMFaithful at long T** (+10pp at T=2048, NLL 0.93 vs 2.12)
- **Level15GSF_NoDrop beats TEMFaithful on sparse landmarks** (TEM's home turf)

The two fixes (NoDrop = remove post-attention residual dropout; GSF = K=8
parallel Kalman chains with mixture weighting) compose. Neither uses TEM
machinery (no per-action W_a, no Hopfield memory).

### Cross-scale and cross-topology generalization

| Variant                          | size 32 | size 64 | size 128 |
| -------------------------------- | ------- | ------- | -------- |
| RoPE                             | 0.423   | 0.525   | 0.522    |
| Vanilla                          | 0.564   | 0.782   | 0.796    |
| Level15GSF_NoDrop (K=8)          | 0.801   | 0.931   | 0.953    |
| **Level15GSF_NoDrop (K=16)**     | **0.971** | **0.992** | **0.998** |

More Kalman chains → better cross-scale generalization (no per-mode ω
parametrization needed; capacity from extra θ_init hypotheses suffices).

### Place cells emerge; grid cells do not

| Variant         | Max grid score | Frac "grid cells" (>0.3) | Frac "place cells" (peak>5×) |
| --------------- | --------------- | ------------------------ | ----------------------------- |
| RoPE            | 0.033           | 0.000                    | 0.555                         |
| Vanilla         | 0.050           | 0.000                    | 0.419                         |
| Level15         | 0.024           | 0.000                    | 0.448                         |
| Level15GSF_NoDrop | 0.025         | 0.000                    | 0.458                         |
| TEMFaithful     | 0.046           | 0.000                    | 0.214                         |

**Place-cell-like units emerge in every architecture trained on
prediction**, including standard transformers (RoPE). Hex-grid cells
do NOT emerge in any architecture — consistent with Sorscher et al. 2019
predicting that hex requires specific objectives (non-negativity + DoG-of-
position targets) absent in standard prediction training.

## Quick start

```bash
pip install -r requirements.txt

# Train a single variant on the standard torus task:
python3 -m mapformer.train_variant \
    --variant Level15 --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 \
    --device cuda --output-dir runs/Level15_clean/seed0

# Reproduce the full multi-seed paper suite (~6h on 2 GPUs):
nohup python3 -u -m mapformer.orchestrator > orchestrator.log 2>&1 &
nohup bash master_finish_v3.sh > master_finish_v3.log 2>&1 &

# Cognitive-map necessity suite (cross-scale, cross-topology, long-T, etc.):
bash run_cognitive_tier.sh

# Zero-shot eval on existing checkpoints:
python3 -m mapformer.zero_shot_eval \
    --runs-dir runs --config lm200 \
    --variants Vanilla VanillaEM Level15 Level15GSF_NoDrop TEMFaithful \
    --model-seeds 0 1 2 --n-test-seeds 5 \
    --lengths 128 512 1024 2048 \
    --output ZERO_SHOT_TRANSFER_lm200.md
```

`train_variant.py --variant` accepts: `Vanilla`, `VanillaEM`, `Level1`,
`Level15`, `Level15Beta`, `Level15NoDrop`, `Level15GSF`, `Level15GSF_NoDrop`,
`Level15GSF_NoDrop_K16`, `Level15EM`, `Level2`, `PC`, `RoPE`, `LSTM`, `CoPE`,
`MambaLike`, `TEMFaithful`, plus 4 Level15 ablations. See `train_variant.py`
for the complete list.

## Models (brief)

The repository implements paper-faithful MapFormer variants + a family of
extensions sharing the same parallel-scan path integration. The two main
recommended variants:

| Class                                   | File                              | Description |
| --------------------------------------- | --------------------------------- | ----------- |
| `MapFormerWM_Level15GSF_NoDrop`         | `model_inekf_gsf_nodrop.py`       | **Recommended for landmarks / sparse-cue tasks.** K=8 Gaussian Sum Filter + post-attention residual dropout removed. |
| `MapFormerWM_Level15GSF_NoDrop_K16`     | `model_inekf_gsf_nodrop.py`       | **Recommended for cross-scale and cross-topology.** Same as above but K=16. More chains → more diverse initial-position hypotheses → better scale-OOD. |

The full model table (paper-faithful, Level1/1.5/2, PC variants, ablations,
baselines) is in [DETAILED_RESULTS.md §Models](DETAILED_RESULTS.md#models).

## Architecture in 60 seconds

MapFormer's path integration: each action token contributes a learned
angular increment, the cumulative sum gives a per-block angle (multiple
frequencies), and attention uses these angles to RoPE-rotate Q and K.

```
Δ_t       = action_to_lie(x_t)         # (H, NB) per-step angular increments
θ_path_t  = ω · cumsum(Δ)_t            # path-integrated angle, parallel scan
Q, K      = RoPE(Q, θ_path), RoPE(K, θ_path)
attention = softmax(Q · K^T / √d) · V
```

Level 1.5 InEKF adds a bounded-error correction:

```
z_t       = π · tanh(measure_head(x_t))      # measurement
R_t       = exp(log_R_head(x_t))             # measurement variance
K_t       = Π / (Π + R_t)                    # Kalman gain
ν_t       = atan2(sin(z_t − θ_path_t), cos(z_t − θ_path_t))   # wrapped innovation
d_t       = (1 − K_t)·d_{t−1} + K_t·ν_t      # affine scan, parallel-scannable
θ̂_t       = θ_path_t + d_t                   # corrected angle
```

The wrap on innovation is what keeps `θ̂` bounded at OOD sequence length —
the key mechanism for length extrapolation.

Gaussian Sum Filter extends this with K parallel chains differing in
their initial position hypothesis, weighted by cumulative log-likelihood.
See [DETAILED_RESULTS.md §"Why Level 1.5 wins"](DETAILED_RESULTS.md) and
[DETAILED_RESULTS.md §Key math](DETAILED_RESULTS.md) for the full
mathematical derivations.

## Critical implementation invariants

Do not regress these (each was a debugging session):

1. **Torus grid:** `(x+dx) % N`, not `clip(x+dx, 0, N-1)`.
2. **Unified token stream:** actions and observations share one embedding
   table; the model must *learn* which tokens update position vs content.
3. **Revisit-only loss:** first-visit observations are random; training on
   them collapses the model to "always predict blank."
4. **ω init is monotonically DECREASING in i:** `ω_i = ω_max · (1/Δ_max)^(i/(n_b−1))`.
   (Paper eq. 17 has a sign typo.)
5. **EM attention is Hadamard `softmax(A_X ⊙ A_P)·V`**, not additive.
6. **MapEM uses separate learnable `q₀ᵖ` and `k₀ᵖ`**, both rotated by the
   path-integrated angle.
7. **Low-rank Δ projection** `W_Δ = W_Δ^out · W_Δ^in` with bottleneck `r=2`.
8. **InEKF measurement head is content-only.** Adding `(cos θ̂, sin θ̂)`
   creates a degenerate optimum.
9. **Wrap innovations modulo 2π** via `atan2(sin(z−θ̂), cos(z−θ̂))`.
   Without this, length generalization breaks.
10. **`log_R_init_bias=3.0` for any backbone whose attention has no
    fallback path** if the position branch is corrupted at init
    (notably MapFormer-EM). See
    [DETAILED_RESULTS.md §"Init pathology"](DETAILED_RESULTS.md).

## Project structure

```
mapformer/
  ── core ──────────────────────────────────────────────────────────────
  environment.py             Torus GridWorld; interleaved tokens; revisit mask
  environment_multienv.py    Multi-env wrapper (same-class instances)
  environment_multisize.py   Multi-scale wrapper (32/64/128 torus mixed)
  environment_topology.py    Multi-topology wrapper (torus + open + walls)
  environment_goal.py        Goal-directed GridWorld (BFS-supervised)
  minigrid_env.py            MiniGrid wrapper (DoorKey, MultiRoom, etc.)
  lie_groups.py              SO(n) exp/log; block-diagonal rotations
  prefix_scan.py             O(log T) parallel prefix product

  ── models ──────────────────────────────────────────────────────────────
  model.py                   MapFormerWM, MapFormerEM (paper-faithful)
  model_inekf_*.py           Level 1, 1.5, 2 InEKF variants
  model_inekf_level15_nodrop.py  Level 1.5 with post-attn dropout removed
  model_inekf_gsf*.py        Gaussian Sum Filter (multi-modal Bayes)
  model_inekf_level15_em.py  Level 1.5 on MapFormer-EM backbone
  model_tem_faithful.py      TEM baseline (per-action W_a + Hopfield memory)
  model_baseline_rope.py     Standard RoPE (cognitive-map necessity baseline)
  model_baselines_extra.py   LSTM, CoPE, MambaLike

  ── training and eval ──────────────────────────────────────────────────
  train_variant.py           Unified single-variant trainer (used everywhere)
  train_multienv.py          Multi-env trainer
  train_multisize.py         Multi-scale trainer
  train_topology.py          Multi-topology trainer
  orchestrator.py            Main multi-seed paper experiment suite
  zero_shot_eval.py          Three-axis OOD eval
  long_sequence_eval.py      Eval at T up to 2048
  calibration_analysis.py    ECE + reliability diagrams
  eval_long_t.py             Cognitive-map necessity long-T eval
  eval_ood_grid.py           OOD-d / OOD-s grid generalization eval

  ── probes / diagnostics ───────────────────────────────────────────────
  probe_goal_linear.py       Frozen cognitive-map → action linear probe
  probe_vector_nav_v2.py     Pair-based Tolman vector-navigation probe
  probe_hex_emergence.py     Place / grid cell rate-map analysis
  probe_gsf_modes.py         GSF mode-weight diagnostic
  probe_modeomega_lesion.py  Mode-omega capacity vs specialization disambiguation

  ── outputs (auto-generated) ───────────────────────────────────────────
  RESULTS_SUMMARY_2026-05-10.md   Session walkthrough (PDF available)
  DETAILED_RESULTS.md             Full historical record
  RESULTS_PAPER.md                Paper-ready multi-seed tables
  *_RESULTS.md                    Per-experiment results
  paper_figures/                  Length-gen + calibration + ablation PNGs
  runs/{Variant_Config}/seed{N}/{Variant}.pt   Multi-seed checkpoints
```

## Honest framing

MapFormer (the paper) already solves the aliased 2D-torus next-token
prediction task it introduces. **We are not beating the paper on the
paper's task.** Our contribution is to:

1. Show MapFormer's cognitive-map inductive bias is *necessary* across
   six independent cognitive demands (RoPE collapses; MapFormer wins).
2. Identify two architectural improvements (NoDrop, GSF) that match or
   beat TEMFaithful — the strongest existing cognitive-map baseline —
   on lm200, long sequences, sparse landmarks, and multi-environment
   generalization.
3. Extend MapFormer to cross-scale and cross-topology generalization
   without architectural specialization (just more Kalman chains).
4. Probe for hippocampal-like representations: place-cell-like units
   emerge universally; hex-grid cells do not.

What we explicitly did NOT do (and acknowledge as limitations):
- Closed-loop goal-directed behavior is near-chance on torus (BC
  distribution-shift ceiling). DAgger on DoorKey lifts Level15NoDrop
  from 0.24 to 0.42; on torus closed-loop training was not pursued
  to completion.
- SR readout did not produce reward-conditional planning (design
  flaw with sparse MSE targets).
- Cross-class transfer (torus + MiniGrid) was attempted; results
  are in `MULTICLASS_RESULTS.md`.
- True RL on top of the cognitive map is out of scope.

## References

- **Rambaud, Mascarenhas, Lakretz (2025).** *MapFormer: Self-Supervised
  Learning of Cognitive Maps with Input-Dependent Positional Embeddings.*
  [arXiv:2511.19279](https://arxiv.org/abs/2511.19279)
- **Whittington et al. (2020).** *The Tolman-Eichenbaum Machine.* Cell.
- **Barrau, Bonnabel (2017).** *The Invariant Extended Kalman Filter as a
  Stable Observer.* IEEE TAC 62(4).
  [arXiv:1410.1465](https://arxiv.org/abs/1410.1465)
- **Marković, Ćesić, Petrović (2017).** *On wrapping the Kalman filter
  and estimating with the SO(2) group.*
  [arXiv:1708.05551](https://arxiv.org/abs/1708.05551)
- **Särkkä, García-Fernández (2021).** *Temporal Parallelization of
  Bayesian Filters and Smoothers.* IEEE TAC 66(1).
- **Alspach, Sorenson (1972).** *Nonlinear Bayesian Estimation Using
  Gaussian Sum Approximations.* IEEE TAC 17(4).
- **Sorscher et al. (2019).** *A unified theory for the origin of grid
  cells through the lens of pattern formation.* NeurIPS.
- **Stachenfeld et al. (2017).** *The hippocampus as a predictive map.*
  Nat. Neuroscience.
- **Banino et al. (2018).** *Vector-based navigation using grid-like
  representations in artificial agents.* Nature.
- **Su et al. (2021).** *RoFormer: Enhanced Transformer with Rotary
  Position Embedding.*
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Gu, Dao (2024).** *Mamba: Linear-Time Sequence Modeling.*
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **Ramsauer et al. (2021).** *Hopfield Networks is All You Need.* ICLR.
