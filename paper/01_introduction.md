# 1. Introduction

MapFormer (Rambaud et al., 2025) is the current state-of-the-art
Transformer architecture for learning cognitive maps of structured
environments. By replacing RoPE's fixed positional rotations with
*input-dependent* rotations driven by the action stream, MapFormer
performs path integration in the Lie algebra of SO(n) via cumulative sum,
preserving Transformers' O(log T) parallelism while recovering TEM-style
(Whittington et al., 2022) structural disentanglement. On the authors'
aliased-observation next-token prediction benchmark, MapFormer-EM reaches
0.999 in-distribution accuracy and MapFormer-WM reaches 0.955; neither
leaves meaningful headroom on that specific task.

**The authors' task is therefore solved.** A paper that claimed
"we improve on MapFormer's own task by an additional fraction of a
percentage point" would be uninteresting, and we do not make that claim.
Instead, this paper takes MapFormer's benchmark as a confirmed starting
point and asks whether the surrounding architectural neighbourhood holds
larger, currently-unrealised wins:

1. **Action noise**, which the paper does not study. Real navigation is
   noisy; if the path-integration estimate drifts, what does the
   architecture do?

2. **True (non-aliased) landmarks**, also not studied in the paper. The
   paper's observation vocabulary has 16 symbols mapped randomly to 4096
   grid cells, so every observation is by construction ambiguous.
   Environments with a small fraction of uniquely-identifying landmarks
   should let the model snap back to a precise position estimate — but
   only if it has a mechanism for doing so.

3. **Out-of-distribution sequence length**, tested only to T=512 in the
   paper. Classical Kalman filtering has provable bounded-error
   guarantees under observability; do MapFormer-family architectures with
   explicit state correction inherit these guarantees empirically?

4. **Calibration**, not measured in the paper. Point accuracy can saturate
   without the model knowing *when* to be confident. A model that outputs
   per-token uncertainty should be able to sharpen its distribution at
   informative tokens and broaden it elsewhere.

**Our contribution** is a family of parallel Invariant Extended Kalman
Filter (InEKF, after Barrau & Bonnabel 2017) extensions to MapFormer that
add explicit state-correction machinery without sacrificing O(log T)
scan depth. We build on **MapFormer-WM** (the weaker of the paper's two
variants) because its single-branch input-dependent rotation admits a
clean coupling between the filter's corrected state θ̂ and the attention
stream; a port to MapFormer-EM is mechanically straightforward but not
explored here. We report multi-seed MapFormer-EM numbers as an
upper-bound reference throughout.

**Four findings.**

1. **A fully-parallel InEKF on SO(2) is practical.** The closed-form
   scalar DARE yields a steady-state gain implementable as an FFT-based
   convolution (Level 1); adding a per-token learned measurement noise
   R_t yields time-varying gain implementable as a scalar Hillis-Steele
   affine scan (Level 1.5); a full heteroscedastic prior Π_t requires a
   Möbius-matrix scan (Level 2). All three match the parallelism profile
   of Mamba-family selective SSMs.

2. **Simpler filters win.** Level 2 is strictly more expressive than
   Level 1.5 yet performs measurably worse in our experiments. We
   diagnose this as optimisation difficulty from the Möbius scan, since
   replacing it with a learnable scalar while retaining the
   heteroscedastic gain (Level 1.5) recovers all the benefits at a
   fraction of the compute. One ablation (Level 1.5-DARE) goes further
   and fixes Π from the scalar DARE rather than learning it — and is
   competitive or slightly better on the landmark regime.

3. **On the paper's clean task, Level 1.5 essentially matches
   MapFormer-WM.** This is the honest baseline comparison: 1.000 vs 0.990
   at T=128, 0.995 vs 0.921 at T=512 (4× OOD), with NLL 0.000 vs 0.031 —
   the NLL gap reflects calibrated confidence at landmark tokens rather
   than a raw accuracy gap. The clean task was not where the contribution
   was expected to live, and it is not where we claim it.

4. **In the paper's untested regimes, Level 1.5 opens measurable gaps.**
   Under 10% action noise at T=512 (4× OOD), Level 1.5 beats
   MapFormer-WM by +11 pp (0.851 vs 0.739). With 200 true landmarks
   added to the grid at T=512 OOD, Level 1.5 again beats MapFormer-WM by
   +11 pp (0.821 vs 0.715). Both differences hold across three training
   seeds and are accompanied by roughly 2× lower NLL, matching the
   classical filtering story: bounded error under observability, sharper
   posteriors at informative tokens.

**What this work is not.** We are not introducing a new benchmark; we
are not scaling to large models; and we are not claiming to have beaten
a state-of-the-art that was at 0.99+ before we started. We are
demonstrating that a classical robotics tool (the Invariant EKF),
parallelised to match MapFormer's depth profile, extends MapFormer's
reach into regimes the paper acknowledged it did not test. We view this
as a first step along a transfer direction — classical Lie-group
robotics machinery into deep learning architectures — that remains
underexplored; Section 6 lays out the natural follow-ups, including
SE(n) generalisations, IMU-preintegration-style context compression,
and SE(n)-equivariant attention.
