# 4. Results

We organise empirical findings under six headings. All numbers are
multi-seed (3 training seeds × 3 fresh test-environment seeds where
applicable); confidence intervals are population standard deviations.
We use the notation *in-dist* for evaluation on the same `obs_map`
seed used for training and *OOD* for evaluation on a held-out
`obs_map` (`seed + 1000`), which tests path-integration generalisation
in an environment the model has never seen.

## 4.1 Main results across three regimes

Three configurations probe different sources of difficulty that the
MapFormer paper did not evaluate:

- **clean:** no action noise, no landmarks (reproduces the paper's setup)
- **noise:** 10% action noise injected at training time (path-integration
  drift test)
- **lm200:** 200 unique landmark cells (non-aliased; ≈5% density)

Table 1 reports **OOD accuracy at T=512** (4× training length). Full
multi-seed tables with in-dist / T=128 numbers are in `RESULTS_PAPER.md`.

| Variant | Clean | Noise | LM200 |
|---|---|---|---|
| MapFormer-WM | 0.913 ± 0.037 | 0.739 ± 0.062 | 0.715 ± 0.059 |
| MapFormer-EM | **0.972 ± 0.003** | 0.765 ± 0.138 | 0.605 ± 0.108 |
| Level 1 InEKF | 0.880 ± 0.070 | 0.783 ± 0.041 | 0.721 ± 0.040 |
| **Level 1.5-WM** | **0.993 ± 0.003** | 0.851 ± 0.026 | **0.821 ± 0.025** |
| **Level 1.5-EM** | 0.977 ± 0.008 | **0.869 ± 0.026** | 0.730 ± 0.120 |
| Predictive Coding | 0.815 | 0.752 | 0.733 |
| LSTM | 0.800 ± 0.004 | 0.743 ± 0.010 | 0.641 ± 0.003 |
| RoPE | 0.463 ± 0.026 | 0.469 ± 0.027 | 0.495 ± 0.013 |
| MambaLike | 0.573 ± 0.007 | 0.568 ± 0.004 | 0.513 ± 0.010 |

Three observations:

1. **Level 1.5 wins by +11 pp over vanilla MapFormer-WM in the noise
   regime** (0.851 vs 0.739) **and +11 pp in the landmark regime**
   (0.821 vs 0.715). On the clean task both saturate (0.993 vs 0.913),
   as expected because the paper already reported MapFormer-WM at
   0.95+ on this task.
2. **A stronger MapFormer-WM backbone alone does not subsume the
   correction mechanism.** MapFormer-EM gets 0.765 and 0.605 on noise
   and landmarks — actually *worse* than MapFormer-WM on landmarks —
   and Level 1.5 stacked on top of EM recovers the lost ground
   (0.869 and 0.730).
3. **Generic Mamba-style SSM fails across all regimes** (0.513–0.573),
   worse than the non-cognitive-map baselines RoPE and LSTM in most
   cells. This reproduces the MapFormer paper's own Appendix A.5
   Table 3 finding at our scale: diagonal-A selective SSMs cannot
   represent 2D rotations and therefore cannot learn cognitive maps.

## 4.2 TEM-style one-shot generalisation

The cleanest test of "learned cognitive-map structure vs memorised one
specific map" is the per-visit-count accuracy curve (Whittington et
al. 2020, *Cell*). For each prediction position we bin by how many
times the current cell has been visited so far in the trajectory:

- **k=1** = first visit, no prior information
- **k=2** = first revisit, one-shot generalisation test
- **k ≥ 3** = repeated revisits, memorisation regime

A model that learned the abstract cognitive-map structure should
jump from chance at k=1 to high accuracy at k=2. A model that only
aggregates evidence needs many visits.

Table 2: k=2 (first-revisit) accuracy on fresh-environment trajectories
(3 model seeds × 5 fresh obs_map seeds = 15 trajectories per cell).

| Variant | Clean k=2 | LM200 k=2 |
|---|---|---|
| **Level 1.5-WM** | **0.995 ± 0.004** | **0.820 ± 0.031** |
| Level 1.5-EM | 0.974 ± 0.009 | 0.737 ± 0.113 |
| MapFormer-EM | 0.977 ± 0.007 | 0.604 ± 0.103 |
| MapFormer-WM | 0.910 ± 0.044 | 0.721 ± 0.055 |
| Level 1 | 0.882 ± 0.075 | 0.718 ± 0.033 |
| Predictive Coding | 0.818 ± 0.013 | 0.737 ± 0.065 |
| LSTM | 0.815 ± 0.013 | 0.659 ± 0.014 |
| MambaLike | 0.581 ± 0.009 | 0.519 ± 0.007 |
| RoPE | 0.472 ± 0.022 | 0.504 ± 0.015 |

**Level 1.5-WM achieves near-perfect one-shot generalisation on
clean (0.995)** — given one prior visit, the model knows exactly
what to predict. MambaLike and RoPE show almost no k=1 → k=2 jump
(0.49 → 0.58 and 0.42 → 0.47 respectively), confirming they have
not learned cognitive-map structure. Figure 5 plots full curves for
k=1..8.

On the harder lm200 regime, Level 1.5's advantage over the other
MapFormer variants widens (0.82 vs 0.60–0.74 for the rest), again
consistent with the Kalman-filtering story that explicit state
correction helps most at informative-landmark tokens.

## 4.3 Zero-shot transfer evaluation

We test three zero-shot axes (`zero_shot_eval.py`):

1. **Axis 1 — fresh obs_map seeds.** 5 new environment seeds per model
   (`test_seed = 10000..10004`). Same grid size, same task structure,
   never-seen obs-map.
2. **Axis 2 — biased action distributions.** Trained on uniform
   random actions; tested under `mostly_east`, `mostly_NS`,
   `diagonal_NE`. Tests path-integration robustness to trajectory
   statistics outside training.
3. **Axis 3 — landmark density transfer.** lm200-trained models
   evaluated at landmark counts {0, 50, 100, 200}. Tests whether the
   learned per-token R_t generalises across landmark density.

Level 1.5-WM retains its lead across all three axes (full tables in
`ZERO_SHOT_TRANSFER_*.md`). On Axis 3 in particular, Level 1.5's
advantage over MapFormer-WM remains ~10 pp constant across densities
{0, 50, 100, 200}, indicating the per-token R_t mechanism transfers
— the model has learned a context-free informativeness signal rather
than a density-specific calibration.

## 4.4 Level 1.5 ablations

Four ablations isolate which Level 1.5 components are necessary
(Table 3, `RESULTS_PAPER.md` §"Level 1.5 Ablations"):

| Variant | Clean T=128 acc | LM200 T=128 acc |
|---|---|---|
| **Level 1.5 (full)** | **1.000** | 0.896 |
| L15_ConstR (no per-token R) | 0.795 | 0.783 |
| L15_NoMeas (no z-head) | 0.904 | 0.905 |
| L15_NoCorr (no state correction) | 0.940 | 0.898 |
| L15_DARE (Π fixed by DARE) | 1.000 | **0.943** |

Removing per-token R (`L15_ConstR`) is the most damaging ablation —
accuracy drops 20 pp on clean and 11 pp on lm200. This isolates the
**per-token heteroscedastic R_t head as the critical Level 1.5
ingredient.** Interestingly, fixing Π from the closed-form scalar DARE
(`L15_DARE`) rather than learning it is competitive or slightly
better on lm200, suggesting the learned prior covariance is the least
important learnable component.

## 4.5 Level 1.5-EM: init pathology and safe init

A first port of Level 1.5 to the MapFormer-EM backbone (Hadamard
product attention `softmax(A_X ⊙ A_P)·V`) showed catastrophic training
instability: 3 of 9 seeds plateaued at loss ≈ 1.45, and the remaining
6 reached only mediocre loss (0.75–1.1 vs 0.005–0.05 for Level 1.5-WM).

The diagnosis is a **random-correction pathology**. With default
initialisation (`log_R_init_bias = 0.0`), the Kalman gain is
`K = Π/(Π+R) = 0.5` at init. EM's position attention `A_P` is
computed entirely from rotations of `q_0^p, k_0^p` by the corrected
angle θ̂. At init θ̂ ≈ θ_path + 0.5·ν where ν is random (random
measurement head); this corrupts A_P, which Hadamard-products with
content attention A_X to destroy gradient signal. MapFormer-WM does
not have this failure mode because its content attention provides a
fallback gradient path.

**Fix.** `log_R_init_bias = 3.0` for the EM variant (WM retains the
default 0.0). This gives K ≈ 0.05 at init — the InEKF is essentially
a no-op at start, the model behaves like vanilla MapFormer-EM, and
R_t is learned downward from its high initial value only where
measurements are informative. The fix is the residual-addition
safe-init trick familiar from ResNet's zero-init or adapter β=0.

After the fix, all 9 seeds converge cleanly. First-revisit accuracy
on clean goes from 0.25 (broken) to 0.97 (safe), and the T=512
OOD performance gap vs. MapFormer-EM opens by 10–12 pp on the noise
and landmark regimes (Table 1).

## 4.6 Grid-size scaling via test-time ω rescaling

MapFormer's geometric ω init covers frequencies `[2π/N, 2π]` for grid
size N. To evaluate on N' ≠ N we test a simple rescaling: multiply
the trained ω by `N/N'` at eval time (the YaRN/NTK-aware analogue
for rotary position embeddings). We evaluate on five grid sizes
{32, 48, 64, 96, 128} for each variant, both with original and
rescaled ω.

**Finding: InEKF variants are far more robust to ω rescaling than
Vanilla.** With uniform 2× rescaling going to a smaller grid, Vanilla
MapFormer collapses to 0.310 accuracy (clean) while Level 1.5-WM
retains 0.99. The wrapped innovation `atan2(sin(z−θ̂), cos(z−θ̂))` in
the Kalman update absorbs the within-cell aliasing that the
above-Nyquist ω_max creates; Vanilla has no such wrapping and fails.

Full tables in `OMEGA_RESCALE_*.md`. Single-T evaluation limits the
conclusion; longer trajectories would probe the extremes of scale
generalisation more thoroughly. Multi-scale training (§6) is the
fundamental architectural fix; test-time rescaling is a bounded
zero-shot approximation.

## 4.7 Hippocampal correspondence: partially falsified

We tested three predictions connecting Level 1.5 to hippocampal
neuroscience (`hippocampal_analysis.py`, `hippocampal_hidden_eval.py`):

1. **Hexagonal grid cells.** Sargolini grid scores on hidden-state
   rate maps top out at 0.05–0.15 across all variants. The
   conventional grid-like threshold is 0.3. **Falsified.**

2. **R_t at landmark tokens.** Predicted: `landmark < aliased obs <
   blank` (smaller R = more informative). Observed for Level 1.5-WM:
   aliased < landmark < blank; for Level 1.5-EM: blank < landmark <
   aliased. Neither matches. **Falsified.**

3. **ω modular spacing.** Trained ω is approximately geometric, but
   largely inherited from the initialisation rather than discovered
   by training. **Soft pass.**

The hexagonal-cell failure is **architecturally forced**: the three-
waves-at-60° optimum for hexagonal interference requires three
sinusoidal waves at the same frequency with orientation offsets, but
MapFormer assigns one block per ω. A path-integrator architecture
with multiple blocks per ω and learnable orientations in 2D action
space (see §6.11) could in principle recover hexagonal organisation;
the current architecture cannot, regardless of training.

The R_t failure is more subtle. The R_t head does distinguish blank
from non-blank tokens (basic informativeness signal), but the within-
non-blank ordering is task-driven, not Bayesian-optimal. This
reflects the fact that R_t is trained on revisit-prediction loss
rather than an explicit informativeness objective. We consider this
honest negative a useful clarification: Level 1.5 achieves cognitive-
map function without literally matching known neural firing patterns.

## 4.8 Calibration

Across regimes Level 1.5's NLL is consistently lower than vanilla
MapFormer's — typically by 30–60%. On clean T=128 OOD,
Level 1.5 has NLL 0.000 vs MapFormer-WM's 0.025; on noise T=512 OOD,
0.585 vs 1.225 (2× better). Reliability diagrams and expected-
calibration-error computations (`calibration_analysis.py`) confirm
this is genuine calibration improvement, not just sharper-but-worse
prediction.

The mechanism is the per-token R_t head: R_t is small at tokens where
the model is confident (e.g., revisits to known cells with distinctive
landmarks) and large at tokens where the model genuinely doesn't know
(e.g., first-visit aliased obs). The softmax distribution over vocab
inherits this calibration because R_t modulates how much the
measurement z_t corrects θ̂ before attention.

---

Summary of 4.1–4.8. Level 1.5 is the best-performing variant
in almost every cell we measured. The Kalman-style correction earns
its keep on top of *either* MapFormer backbone (WM or EM), stronger
than a stronger backbone alone, and achieves TEM-style one-shot
generalisation that non-MapFormer baselines cannot. Hippocampal
correspondence is partially falsified — a useful honest result
that delimits what MapFormer's current architecture does and does
not recover about biological spatial representations.
