# 5. Discussion

## 5.1 What Level 1.5 InEKF is, and what it is not

Level 1.5 is a parallel Invariant Extended Kalman Filter on SO(2),
stacked on top of MapFormer's existing path-integration scan. It adds
three learnable components — a constant prior covariance Π, a
per-token measurement noise R_t, and an inverse measurement model
z_t — connected by the scalar Hillis-Steele affine scan

$$d_t = (1 - K_t) d_{t-1} + K_t \cdot \mathrm{wrap}(z_t - \hat\theta_{\text{path},t}), \quad \hat\theta_t = \hat\theta_{\text{path},t} + d_t$$

with Kalman gain $K_t = \Pi / (\Pi + R_t)$. The scan has the same
O(log T) depth as MapFormer's underlying cumsum and the same depth
as Mamba's selective-scan primitive.

**What it is.** A specialised 1-D state-space model with Kalman
parameterisation, Lie-group-correct wrapping, and input-dependent
measurement noise. The parameterisation makes R_t interpretable as
per-token measurement uncertainty by construction — its value has a
direct probabilistic meaning, not merely a learned scaling.

**What it is not.** Level 1.5 is not a generic selective-scan block
that happens to work. Its structure is specifically chosen to satisfy
three requirements: (i) path-integration on SO(2) with the Marković
et al. (2017) wrapped-innovation correspondence to the Lie-Group EKF,
(ii) state correction via an inverse measurement model that respects
the group structure, and (iii) compatibility with MapFormer's
existing geometric ω allocation. Removing any of these via ablation
(`L15_NoMeas`, `L15_NoCorr`, `L15_ConstR`) degrades performance
measurably.

## 5.2 Relation to adjacent work

**MapFormer (Rambaud et al. 2025).** We reproduce MapFormer faithfully
(seven implementation details in §3 and the README) and inherit its
input-dependent path integration. Our contribution is orthogonal to
MapFormer's: it adds state correction where MapFormer had none. On
the paper's benchmark (aliased 2D torus next-token prediction), both
saturate at near-ceiling accuracy, so our contribution lives in
regimes the paper did not test — noise, landmarks, OOD length,
calibration.

**TEM / TEM-T (Whittington et al. 2020, 2022).** TEM is the closest
model family conceptually: it also learns cognitive-map
representations via action-conditioned recurrence. Our per-visit-count
curves (§4.2) are the TEM-canonical one-shot generalisation test,
and Level 1.5 matches or exceeds TEM's qualitative signature
(sharp k=1 → k=2 jump). TEM trains on multiple environments
simultaneously and compares representations to neural data at
length; we compare only at the ω-modular-spacing level, and
candidly report negative results for grid-cell and boundary-cell
correspondence (§4.7).

**Mamba and selective SSMs (Gu & Dao 2023).** The connection is
deep. Wang et al. (2025) show Mamba's selective SSM is isomorphic to
a time-varying linear Kalman filter; Level 1.5's scan is a 1-D Mamba
block with three specific constraints (Kalman-ratio gain, Lie-group
wrap, decoupled path-integration + correction). The key empirical
finding is that this specialisation matters: our `MambaLike` baseline
(generic selective SSM at the same scale) fails to learn the
cognitive-map task (accuracy ≈ 0.57 across configs), reproducing the
MapFormer paper's own Appendix A.5 Table 3 result. Generic SSMs can
in principle approximate this structure but cannot discover it from
gradient descent at our training scale; the explicit Lie-group prior
earns its keep.

**Parallel Bayesian filters (Särkkä & García-Fernández 2021;
Yaghoobi et al. 2021).** Our parallel InEKF is a specific instance of
this broader program — specialised to SO(2) with a scalar affine
scan rather than the general matrix-associative primitive.

**Contextual Positional Encodings (CoPE, Golovneva et al. 2024;
TAPE, Zhu et al. 2025).** These methods address a different problem:
making position encoding content-dependent at the token level. They
do not implement path integration on a compact group, and per the
MapFormer paper's Table 2 both fail on the 2D navigation task
(CoPE 0.74 IID, TAPE 0.23 IID vs MapFormer ≈ 1.00). They are
complementary rather than competing; a CoPE-style content-based
gating layered on top of MapFormer could in principle provide
landmark-detection gating for the R_t head (future work).

## 5.3 Honest limitations

**Single-environment training.** Each model trains on one fixed
`obs_map` (paper-standard for MapFormer). Our OOD evaluations use
fresh obs_maps and show ceiling-level zero-shot transfer to them,
but we do not do TEM-style multi-environment training where the
model sees thousands of distinct environments. The additional
robustness such training would provide is conjectural on our data;
it is a natural follow-up (§6).

**Scale.** We follow MapFormer's 1-layer, 2-head, d=128 configuration.
We have not scaled model or data meaningfully. A comparison paper at
4 layers and 1M+ environments would change the story in unknown ways.

**Single Lie group.** Level 1.5 is developed for SO(2), MapFormer's
native group. The generalisation to SE(2), SE(3), or higher
dimensions involves the same parallel-filter machinery but requires
Adjoint-weighted recurrences (Solà et al. 2018) rather than scalar
affine scans. We describe the path but have not implemented it.

**Hippocampal correspondence.** As §4.7 reports, our current
architecture does not produce hexagonal grid cells (1 block per ω is
structurally insufficient for the three-waves-at-60° interference
required), and R_t does not match the predicted Bayesian-optimal
pattern. §6.11 proposes an architectural modification (MapFormer-Grid)
that could recover hexagonal organisation; testing whether that
actually works is left for follow-up.

**Small Level15-EM variance.** One lm200 seed reached a marginally
worse final loss (1.40 vs ~1.0), inflating the lm200 std to ±0.12.
We report this honestly rather than masking it; tightening would
require more aggressive safe init (`log_R_init_bias = 5.0` or
warmup scheduling).

**CoPE baseline is incomplete.** Due to CoPE's high per-epoch cost
(~500 min/run vs LSTM's 8 min/run on our setup), the lm200 CoPE row
has 2 of 3 seeds. Our other baselines (MambaLike, LSTM, RoPE) are
all multi-seed complete.

## 5.4 When to use Level 1.5 vs vanilla MapFormer

Three rules of thumb emerge from our ablations and per-regime
results:

1. **If the environment is noise-free and fully aliased, MapFormer
   is already at ceiling.** Level 1.5's gain at T=128 clean is
   within noise of the baseline. Use Level 1.5 only if you need
   calibrated uncertainty (the NLL gap is still substantial).
2. **If actions are noisy or the environment has landmarks, Level 1.5
   helps substantially** (+11 pp at T=512 OOD in both regimes).
   This is the primary use case.
3. **If you need calibrated uncertainty for downstream planning**
   (e.g., active navigation), Level 1.5 is strictly preferable even
   on clean tasks. Its per-token R_t provides explicit confidence
   that vanilla MapFormer cannot.

## 5.5 Broader implication: Mamba-as-Kalman on a Lie group

A broader reading of our work: it is **a specialised
Mamba-style selective-scan on SO(2)**. The Wang-et-al. view (Mamba =
linear Kalman filter) tells us that selective SSMs *can* implement
Kalman filters. Level 1.5 shows that they *should*, at least on
tasks with known geometric structure, because the specialisation
(Kalman-ratio gain, Lie-group wrapping, decoupled path-integration
scan) yields a large improvement in what the model can discover from
gradient descent at limited training scale.

This suggests a broader research direction: **constrained Mamba
architectures with explicit group-equivariance**. SO(2) is the
simplest case; SE(2) and SE(3) are natural generalisations for 2D-
and 3D-navigation tasks; other compact Lie groups cover other
periodic structures. In each case, the specialised Mamba block would
look like Level 1.5 but with an Adjoint-weighted scan and
higher-dimensional measurement model. We expect the gradient-descent
discoverability advantage to persist across all of them: *generic
selective SSMs cannot discover geometric structure at small scale;
explicit group-aware selective SSMs can.*

This reframing connects our paper to three adjacent literatures:
(i) equivariant architectures, where group structure is specified
rather than learned; (ii) classical robotics, where Lie-group
filtering has been standard since Barrau & Bonnabel; (iii) modern
SSMs, where the Mamba-Kalman correspondence is a recent but deep
observation. Level 1.5 is a small concrete bridge; the broader
synthesis is what we consider the most interesting open direction.

## 5.6 Conclusion

We started from a reproduction of MapFormer, added a parallel
Invariant EKF machinery preserving its O(log T) parallelism, and
characterised the combined architecture across three untested
regimes. The main concrete contribution — Level 1.5 — provides
bounded-error length generalisation, robustness to action noise,
strong landmark utilisation, and calibrated uncertainty, at the cost
of a single learnable Π, one per-token MLP for R_t, one inverse
measurement MLP, and a scalar affine scan. It is mechanistically
simple, architecturally minimal, and empirically effective.

We are not claiming to solve a task MapFormer could not. We are
claiming that classical Lie-group filtering, parallelised, is a
natural extension of MapFormer-family architectures into precisely
the regimes that any deployed navigation system must handle. The
transfer direction — classical robotics → deep learning, via explicit
group-aware primitives — remains underexplored, and this paper is
one step along it.
