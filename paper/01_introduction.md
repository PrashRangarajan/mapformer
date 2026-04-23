# 1. Introduction

Transformer-based models of spatial and relational reasoning have recently been
shown to learn **cognitive maps** — position-invariant representations of
structured environments that support transfer to novel observations (Whittington
et al., 2022). MapFormer (Rambaud et al., 2025) is the state-of-the-art
architecture in this line: by replacing RoPE's fixed position rotations with
input-dependent rotations driven by the action stream, it performs path
integration in the Lie algebra of SO(n) via cumulative sum — preserving
Transformers' O(log T) parallelism while gaining TEM-style structural
disentanglement.

Yet MapFormer has no explicit mechanism for **state correction**. When the
path-integration estimate drifts (due to noisy actions, imperfect learned
weights, or long horizons beyond attention's window), the model has no way to
exploit external measurements to snap back. Classical navigation filters —
notably the Invariant Extended Kalman Filter (InEKF; Barrau & Bonnabel 2017)
— provide exactly this capability, with provable bounded-error guarantees under
observability. However, naïve Kalman integration into a transformer is
sequential, breaking the parallelism that is MapFormer's defining advantage.

**This paper asks:** can we add the state-correction machinery of classical
Lie-group filtering to MapFormer without breaking its parallelism, and does it
help? Four findings:

1. **A fully-parallel InEKF on SO(2)** is implementable in pure PyTorch using
   FFT convolution (constant-gain case) or Hillis-Steele associative scan
   (time-varying gain). Both match MapFormer's O(log T) depth.

2. **A simpler variant — Level 1.5 — dominates.** Level 2's full heteroscedastic
   formulation (learnable time-varying covariance Π_t via Möbius scan) is
   strictly more expressive than Level 1.5 (constant learnable Π, per-token
   R_t), yet performs measurably worse. We diagnose this as optimization
   difficulty added by the Möbius scan and show that the heteroscedastic K_t
   alone suffices.

3. **Empirical bounded-error generalization.** Level 1 and Level 1.5 degrade by
   only 7–8 pp from T=128 (training length) to T=512 (4× OOD), versus 18–26 pp
   for attention-only and predictive-coding variants. This directly mirrors the
   theoretical bounded-error property of classical Kalman filtering.

4. **Forward and inverse models are complementary.** Predictive Coding (forward
   model: position → expected observation) excels on aliased observations where
   multiple positions share an observation. Kalman-family filters (inverse
   model: observation → position) excel when observations are unique
   (landmarks). Neither is universal; both can coexist with attention-based
   retrieval in a single architecture.

Our headline result is that **Level 1.5 InEKF is a drop-in replacement for
vanilla MapFormer** that matches or exceeds it on all regimes we tested:
in-distribution accuracy, out-of-distribution length generalization,
calibration (NLL), noise robustness, and landmark utilization, at the same
wall-clock training cost.
