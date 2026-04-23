# Abstract

MapFormer (Rambaud et al., 2025) solves the clean aliased-observation
cognitive-map task nearly to ceiling (0.99+ next-token accuracy) by
replacing RoPE's fixed positional rotations with input-dependent rotations
in SO(2). The paper's reported task is therefore not where additional
mechanism is needed. This work asks instead: **what happens when we push
MapFormer into regimes the paper did not test — action noise, true
(non-aliased) landmarks, out-of-distribution sequence lengths, and
calibrated uncertainty — and can classical Lie-group filtering help?**

We contribute a family of **parallel Invariant Extended Kalman Filter**
extensions to the MapFormer-WM backbone, implementable using either FFT
convolution (constant gain) or scalar Hillis-Steele associative scan
(time-varying gain), both matching MapFormer's O(log T) scan depth. Our
main variant — **Level 1.5 InEKF** — uses a constant learnable prior
covariance Π and a per-token measurement noise R_t output by a small MLP,
yielding a heteroscedastic Kalman gain K_t = Π / (Π + R_t) computable in a
single affine scan. A fuller Level 2 (time-varying Π_t via Möbius scan) is
strictly more expressive but optimises worse, a negative result we
characterise mechanistically.

**Empirical findings.** On the paper's clean aliased task, Level 1.5
matches MapFormer-WM within noise (1.00 vs 0.99 at T=128, 0.99 vs 0.92 at
T=512 OOD length) and closes the NLL gap (0.000 vs 0.025, reflecting
calibrated confidence at landmark tokens). The architectural wins appear
outside the paper's tested regime: under 10% action noise at 4× training
length, Level 1.5 beats MapFormer-WM by **+11 pp** (0.85 vs 0.74) with
half the NLL; with 200 true landmarks added to the grid, Level 1.5
beats MapFormer-WM by **+11 pp** (0.82 vs 0.72) at T=512 OOD, again with
roughly half the NLL. Four ablations isolate the per-token R_t head as
the critical ingredient, while a "Level 1.5-DARE" simplification
(closed-form Π from the scalar DARE rather than learning it) is
competitive or slightly better on landmarks, suggesting the learned prior
covariance is the least important learnable component.

**Honest framing.** We are not solving a task MapFormer could not already
solve; we are extending MapFormer into regimes where the filtering
literature predicts bounded-error guarantees should matter, and
confirming empirically that they do. The comparison is against
MapFormer-WM (the weaker of the paper's two variants); multi-seed
MapFormer-EM numbers are also reported as an upper-bound reference. All
methods preserve MapFormer's parallel path-integration scan, with
correction machinery implemented in Mamba-family O(log T) primitives. We
treat this work as a stepping stone toward higher-dimensional Lie groups
(SE(2), SE(3)), IMU-preintegration-style context compression, and
SE(n)-equivariant attention — directions at the classical-robotics /
deep-learning interface that remain underexplored.
