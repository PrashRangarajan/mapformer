# 2. Related Work

**Cognitive maps in neural models.** The Tolman-Eichenbaum Machine (TEM;
Whittington et al., 2019) introduced transformer-adjacent models of the
hippocampal formation that factorize "structure" (position) from "content"
(sensory observations). Later work (TEM-t, Whittington et al. 2022) reformulated
this as a transformer with action-dependent position updates. MapFormer
(Rambaud et al., 2025) is the most recent development, replacing TEM-t's
sequential rotation updates with a parallel cumulative-sum Lie-algebra path
integration.

**Transformer position encodings.** RoPE (Su et al., 2023) rotates
queries/keys by angles that are a fixed function of token index; CoPE
(Golovneva et al., 2024) gates rotations by content to compress relative
distance; TAPE (Zhu et al., 2025) updates positional embeddings after attention.
None of these adapt rotations to an external action signal. MapFormer's core
contribution is precisely this adaptation.

**Cognitive-map filtering frameworks.** The Clone-Structured Cognitive Graph
(CSCG; George et al. 2021) uses discrete latent states ("clones") to disambiguate
aliased observations, representing the Bayesian posterior over position as a
categorical distribution. This is a forward-model strategy: given a position,
predict the expected observation distribution. Classical navigation filters
(EKF, UKF, Invariant EKF) use inverse models: given an observation, infer
the implied position. Our analysis frames these as complementary inductive
biases for the correction mechanism.

**Invariant filtering on Lie groups.** Barrau & Bonnabel (2017) introduced the
Invariant Extended Kalman Filter, showing that on matrix Lie groups with
group-affine dynamics the error admits autonomous (state-independent) dynamics,
enabling bounded-error guarantees under observability. Marković et al. (2017)
showed that on SO(2) the wrapped-innovation EKF is mathematically equivalent
to the full Lie-Group EKF, which we exploit for correctness at minimal
implementation cost.

**Parallel Bayesian filtering.** Särkkä & García-Fernández (2021) introduced
parallelizable Bayesian filters via associative scan, enabling O(log T) depth
for linear-Gaussian state-space models. Yaghoobi et al. (2021) extended this
to nonlinear iterated filters. Mamba (Gu & Dao, 2023) brought such scans to
deep learning as a replacement for attention; our Level 2 design uses the same
associative-scan primitive on Möbius 2×2 matrices, and Level 1.5 uses a simpler
scalar variant.

**Predictive coding.** Rao & Ballard (1999) formalized the "prediction error
as signal" view: top-down predictions compared against bottom-up signals give
error signals that drive state updates. Friston's Free-Energy Principle
generalizes this. We implement a predictive-coding variant of MapFormer that
uses an MLP forward model from (cos θ, sin θ) to expected observation embedding,
with prediction error driving an additive state correction.

**State-space language models.** Mamba (Gu & Dao, 2023), S4 (Gu et al., 2022),
and related selective SSMs use associative scans for parallel-friendly
recurrence. Our Level 2 and Level 1.5 reuse the same scan machinery (Möbius
and scalar-affine respectively) within a transformer-style architecture that
also has attention. The combination matches what attention-SSM hybrids have
explored but applied specifically to SO(2) path integration and state
correction.

**Heteroscedastic Kalman filtering.** Learning per-observation measurement
noise is a well-studied idea in classical filtering (notably the "innovation
covariance estimation" literature of the 1970s–80s) and in modern
learned-filter work such as KalmanNet (Revach et al., 2022). Our Level 1.5
differs from these by coupling heteroscedastic R_t with fixed covariance,
implemented via a single associative scan, making it architecturally lighter
than KalmanNet while retaining the key per-token gain adaptation.
