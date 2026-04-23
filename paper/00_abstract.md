# Abstract

We extend MapFormer (Rambaud et al., 2025), a Transformer-based architecture for
learning cognitive maps via input-dependent positional encoding in SO(2), with
a parallel Invariant Extended Kalman Filter that provides explicit state
correction without sacrificing MapFormer's O(log T) scan depth. Our principal
contribution is **Level 1.5 InEKF**, an architecture that combines a constant
learnable covariance with a per-token measurement noise R_t learned from the
input embedding, yielding a heteroscedastic Kalman gain K_t = Π / (Π + R_t)
implementable as a single scalar affine associative scan. Empirically, Level 1.5
matches vanilla MapFormer in wall-clock speed while exceeding it on essentially
every axis we measured: training loss (−18%), accuracy at training length (+1.5
pp) and 8× out-of-distribution length (+2.1 pp), calibration (−43% to −48%
NLL across sequence lengths), noise robustness (+2–5 pp at matched noise), and
landmark utilization (73% vs 1.7%). We additionally characterize five
correction-mechanism variants (Vanilla, Vanilla+noise, Parallel InEKF, PC,
Level 2 full heteroscedastic) across three regimes (aliased-only, action-noise,
true landmarks) and show that forward-model predictive coding and inverse-model
Kalman filtering are complementary — the former excels at aliased aggregation,
the latter at sharp measurements — and that adding time-varying covariance on
top of time-varying gain (Level 2) does not help and in fact hurts, a negative
result we diagnose mechanistically. All methods preserve MapFormer's parallel
path-integration scan; correction machinery is implemented with FFT convolution
(constant gain) or Hillis-Steele associative scan (time-varying gain), matching
the parallelism profile of Mamba-family selective SSMs.
