# 3. Methods

## 3.1 Notation and Setup

We work on the 2D torus GridWorld from Rambaud et al. (2025). An environment
of size N × N (typically N=64) assigns to each cell an observation type from
K + 1 symbols: K regular types plus a blank token B. A fraction p_empty of
cells are blank. Optionally, L additional cells carry unique single-use
landmark tokens (disjoint from the K regular types).

The agent emits an interleaved trajectory
```
s = (a_1, o_1, a_2, o_2, …, a_T, o_T)
```
with actions a_t ∈ {↑,↓,←,→} uniformly sampled over directed walks (k steps
per chosen direction, k ∈ {1, …, 10}) and observations o_t determined by the
obs-map at the agent's current cell after applying a_t. We train on 200K
sequences (L=128 steps, batch size 128, AdamW lr=3e-4, weight decay 0.05).

**Task.** Self-supervised next-token prediction with loss applied only at
revisited observation positions, matching the paper's forced-navigation
task ("predict the upcoming observation each time the agent returns to a
previously visited location").

## 3.2 MapFormer Baseline (reproduction)

A 1-layer Transformer on the unified vocabulary (actions + observations).
For each token x_t, a low-rank projection produces the Lie-algebra increment
Δ_t ∈ R^{H × B}, where H is the number of attention heads and B is the number
of 2×2 rotation blocks per head. Angular velocities ω ∈ R^{H × B} are learnable
with geometric initialization ω_i = ω_max · (1/Δ_max)^((i−1)/(B−1)), i = 1..B.
(The paper's Eq. 17 specifies ω_i = ω_max · (1/Δ_max)^(−i/B); the negated
exponent is a sign typo, confirmed by the paper's own inequality ω_min =
ω_max/Δ_max and RoPE's analogous decreasing schedule.)

Path integration proceeds in the Lie algebra via parallel cumulative sum:
```
θ_t = ω ⊙ cumsum_t(Δ)
```
where ⊙ is elementwise product. The cumulative angles are applied to queries
and keys via RoPE (Eq. 16 of the paper). Attention is standard causal
self-attention.

The correct paper reproduction requires seven specific fixes (torus boundary,
unified action/observation vocabulary, revisit-only loss, ω sign, Hadamard
EM attention, separate q_0^p/k_0^p, cumsum+exp not prefix-product; see
Appendix B). With all fixes, our reproduction matches the paper's reported
95.5% accuracy at T=128.

## 3.3 Level 1 InEKF: Parallel Invariant EKF on SO(2)

The Level 1 variant adds explicit state correction via a Kalman filter.
Following Marković et al. (2017), on SO(2) the wrapped-innovation EKF is
mathematically identical to the Lie-Group EKF, so we operate on an
unbounded θ̂ ∈ R with innovations wrapped to [−π, π] via `atan2(sin, cos)`.

A learned measurement head produces ẑ_t ∈ [−π, π] from the content
embedding:
```
ẑ_t = π · tanh(W_2 · GELU(W_1 · x_t))
```

With constant process noise Q and constant measurement noise R, the scalar
discrete algebraic Riccati equation has the closed-form fixed point:
```
P* = (−Q + √(Q² + 4QR)) / 2
K* = (P* + Q) / (P* + Q + R)
```

With K* frozen, the correction recurrence becomes a scalar affine recurrence
with constant coefficient:
```
d_t = (1 − K*) · d_{t−1} + K* · ν_t,  ν_t = atan2(sin(ẑ_t − θ_path_t), cos(ẑ_t − θ_path_t))
```
which is equivalent to a causal 1-D convolution with kernel
K*·(1−K*)^k, k = 0, 1, 2, … computable via FFT in O(T log T) work / O(log T)
depth. The corrected angles θ̂_t = θ_path_t + d_t are fed to RoPE in place of
θ_path.

## 3.4 Level 2 InEKF: Full Heteroscedastic

Level 2 learns a per-token measurement noise R_t from the content embedding
and tracks a time-varying covariance Π_t. The covariance recurrence under
predict + update is:
```
Π_{t+1} = (Π_t + Q) · R_t / (Π_t + Q + R_t)
```
a Möbius transformation Π_{t+1} = M_t(Π_t) with matrix
```
M_t = [[R_t + Q,  Q · R_t],
       [ 1,        R_t    ]]
```
Möbius transforms compose under matrix multiplication, so a parallel
associative scan of 2×2 matrices yields Π_t for all t in O(log T) depth.
The gain is then K_t = Π_t / R_t (a scalar simplification derivable from
the predict–update chain). The state correction uses a second affine scan
with time-varying α_t = 1 − K_t.

Two parallel 2×2 matrix scans are ~60× slower in wall-clock than Level 1's
FFT convolution at our sequence lengths (autograd through Hillis-Steele
stores log T intermediate tensor sets per scan), despite equivalent
asymptotics.

## 3.5 Level 1.5 InEKF: The Recommended Architecture

Level 1.5 is the compromise: constant learnable Π, per-token R_t.
This retains the per-token Kalman gain adaptation
```
K_t = Π / (Π + R_t)
```
but eliminates the Möbius covariance scan. The state correction
```
d_t = (1 − K_t) · d_{t−1} + K_t · ν_t
```
is a scalar affine recurrence with time-varying coefficient, parallelizable
via a single scalar Hillis-Steele scan: associative combination
```
(α_1, v_1) ⊗ (α_2, v_2) = (α_1 α_2, α_2 v_1 + v_2)
```
yields cumulative affine transforms in O(log T) depth, O(T log T) work (or
O(T) with Mamba's selective-scan kernel).

Level 1.5 adds four learnable components to vanilla MapFormer:
1. `log_Pi ∈ R^{H × B}`: constant covariance
2. `log_R_head`: MLP (d_model → 128 → H·B) producing per-token log R_t
3. `measure_head`: MLP producing the measurement ẑ_t (same as Level 1)
4. No additional parameters in the scan itself

Total: ~50K additional parameters, ~10% more than vanilla MapFormer's 200K.

## 3.6 Predictive-Coding (PC) MapFormer

An alternative correction mechanism based on forward models rather than
inverse. A forward model
```
ô_t = g(cos θ_path_t, sin θ_path_t)
```
(2π-invariant by construction) predicts the observation embedding from
position. The prediction error ε_t = x_t − ô_t is mapped to a Lie-algebra
correction δθ_t via an MLP, then accumulated through a scalar affine scan
with a learned gate. An auxiliary loss ‖ε‖² at observation positions
encourages the forward model to actually model observations rather than
collapse.

## 3.7 RoPE Baseline

Following the MapFormer paper's own baselines, we include a Transformer
with fixed RoPE (position angles as fixed function of token index,
not learned from actions). Otherwise identical architecture to MapFormer.

## 3.8 Ablations

Four Level 1.5 ablations isolate components:
- **L15_ConstR**: constant learnable R (no heteroscedasticity). Tests whether
  per-token R_t matters.
- **L15_NoMeas**: measurement head output fixed to 0, so ẑ = 0 always. Tests
  whether the measurement signal contributes.
- **L15_NoCorr**: correction scan output forced to 0 (d_t = 0). Recovers
  vanilla MapFormer with idle parameters. Sanity check.
- **L15_DARE**: Π computed via scalar DARE from learnable Q and a separate
  learnable R_avg (not learnable Π directly). Tests whether the free learnable
  Π adds expressivity beyond the DARE-prescribed value.
