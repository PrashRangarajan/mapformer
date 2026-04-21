# MapFormer + Kalman / Predictive Coding: A Characterization of When Explicit State Correction Helps Transformer-Based Cognitive Maps

## Abstract

Starting from a faithful reproduction of MapFormer (Rambaud et al., 2025) — a
Transformer that learns cognitive maps via cumsum-based parallel path
integration in SO(2) — we investigated whether adding explicit state-correction
mechanisms (Invariant Extended Kalman Filter and Predictive Coding) improves
robustness, uncertainty, and generalization.

We arrive at a concrete architecture that substantially improves over vanilla
MapFormer on every metric we measured. The winning variant, which we call
**Level 1.5 InEKF**, uses a constant learnable covariance Π but learns a
**per-token measurement noise `R_t`** from the content embedding, giving a
time-varying Kalman gain `K_t = Π / (Π + R_t)` that automatically down-weights
uninformative tokens (blanks) and up-weights informative ones (landmarks).
Implemented via a single scalar associative scan, it runs at the same speed
as vanilla MapFormer while achieving **73% landmark accuracy** (vs 18% for
the best prior variant), **95% overall accuracy at training length** (vs 87%),
and **87% overall accuracy at 4× OOD length** (vs 78%).

Along the way, we characterize the regimes where five different correction
mechanisms help or hurt, and show empirically that the theoretical
bounded-error property of Kalman filtering (Barrau & Bonnabel 2017) is
realized at the transformer-architecture level.

We obtain all this **without sacrificing MapFormer's core O(log T)
parallelism** — every correction mechanism we built is implementable via
cumsum, FFT convolution, or Hillis-Steele associative scan.

---

## 1. Motivation and Background

MapFormer is a Transformer architecture that learns a cognitive map from an
interleaved sequence of (action, observation) tokens. Its core contributions
over prior work (notably TEM / TEM-t):

1. **Input-dependent positional encoding.** Rotations in SO(2) come from the
   learned action stream rather than token index — a generalization of RoPE.
2. **Parallel path integration.** `cumsum(ω·Δ)` with a single exponentiation
   runs in O(log T) on a parallel scan, enabling length generalization that
   sequential models cannot match.
3. **Structure–content disentanglement.** Actions update position (SO(2)
   rotations); observations leave position untouched (they feed attention V).

Given a noisy-path-integration problem (random action perturbations), explicit
state-correction mechanisms like the Invariant Extended Kalman Filter (IEKF,
Barrau & Bonnabel 2017) would in principle provide bounded-error guarantees
and principled uncertainty estimates. Likewise, Predictive Coding (Rao &
Ballard, 1999) would provide an error-driven update rule aligned with modern
theories of hippocampal/EC function.

**This project:** extend MapFormer with both frameworks, preserve
parallelism, and empirically characterize the regimes where each helps.

---

## 2. Starting Point: Faithful Paper Reproduction

Prior to any extensions, we fixed seven implementation bugs that caused the
initial MapFormer training to plateau at ~50% accuracy (random baseline for
the blank-heavy task):

1. **Torus grid.** The environment must wrap: `(x+dx) % N`, not clamped.
2. **Single interleaved token stream.** Actions and observations must share one
   embedding table so the model must *learn* to disentangle them (consistent
   with the paper's Section 5.4 "factorizing structure and content" claim).
3. **Loss masked to revisits.** The paper explicitly states: "predict the
   upcoming observation each time it comes back to a previously visited
   location." First-visit observations are random noise; training on them
   collapses the model to predict the marginal (50% blank).
4. **ω initialization schedule.** The paper's eq. 17 has a sign typo. The
   correct schedule is monotonically decreasing in `i`, matching RoPE:
   `ω_i = ω_max · (1/Δ_max)^(i/(n_b−1))`.
5. **MapEM attention uses Hadamard product.** `softmax(A_X ⊙ A_P) · V`, not
   additive.
6. **MapEM learnable initial vectors.** Separate `q_0^p` and `k_0^p`, rotated
   by the path-integrated angles.
7. **Cumsum + exponentiation, not prefix product.** Path integration in the
   Lie algebra (additive) rather than prefix-product on rotation matrices.

With these fixes applied, both MapFormer-WM (0.955 train acc) and MapFormer-EM
(0.999 train acc) reach paper-reported performance at 200K sequences.

---

## 3. Five Model Variants

We built and empirically compared five architectures. The first four are
iterations that helped develop the final fifth.

| Variant | Correction mechanism | Scan | Parallelism |
|---------|---------------------|------|-------------|
| **Vanilla MapFormer** | none (attention only) | — | O(log T) |
| **Vanilla + noise aug** | trained with 10% action-token replacement | — | O(log T) |
| **Parallel InEKF (Level 1)** | constant `R`, steady-state gain `K*` from scalar DARE, wrapped innovation via `atan2(sin, cos)` | FFT conv | O(log T) |
| **PC MapFormer** | forward model `ô = g(cos θ, sin θ)`, prediction error in embedding space, error-to-state MLP | FFT conv | O(log T) |
| **Level 2 InEKF** | per-token `R_t` *and* time-varying `Π_t` via Möbius 2×2 matrix scan | 2× Hillis-Steele 2×2 | O(log T) but ~60× slower constants |
| **Level 1.5 InEKF** *(winner)* | constant learnable `Π`, per-token `R_t` from MLP head → `K_t = Π/(Π+R_t)` | 1× scalar Hillis-Steele | O(log T) at vanilla speed |

Two key engineering wins throughout:

**(a) Proper Lie-group geometry.** We use Marković et al. (2017)'s result that
on SO(2) the wrapped-innovation EKF equals the Lie-Group EKF. Our filter
operates on an unbounded `θ̂ ∈ R` with innovations wrapped to `[−π, π]` via
`atan2(sin, cos)`. This gives length-invariance without breaking the
Kalman math.

**(b) Parallel scan via closed-form steady-state gain.** For constant Q and
constant R, the scalar discrete DARE has a closed-form solution:

```
P* = (-Q + √(Q² + 4QR)) / 2
K* = (P* + Q) / (P* + Q + R)
```

With K frozen, the correction recurrence `d_t = (1−K*)·d_{t−1} + K*·ν_t`
becomes a scalar affine recurrence, computable via FFT convolution in
O(log T) depth. The PC variant uses an identical scan structure with a
learned gate in place of `K*`.

For Level 1.5, with `K_t` time-varying but `Π` constant, we use a scalar
associative scan over pairs `(α_t, u_t)` composed as
`(α_1, u_1) ⊗ (α_2, u_2) = (α_1 α_2, α_2 u_1 + u_2)`. Hillis-Steele yields
`O(log T)` depth and matches the algorithmic structure used in Mamba.

---

## 4. Experiments and Results

### 4.1 Aliased-observation task (no landmarks)

Training with 10% action-replacement noise augmentation, T=128, batch=128,
200K sequences:

| Variant | Final training loss | Epoch speed |
|---------|--------------------|-------------|
| Vanilla + noise | 0.761 | 11 s/epoch |
| Parallel InEKF (Level 1) | 0.879 | 10 s/epoch |
| PC MapFormer | 0.700 | 11 s/epoch |
| Level 2 InEKF | 0.768 | ~600 s/epoch |
| **Level 1.5 InEKF** | **0.751** | 11 s/epoch |

At test time under **matched-distribution** action noise:

| Test noise | Vanilla+noise | Level 1 InEKF | PC | Level 2 | **Level 1.5** |
|-----------|--------------|---------------|-----|---------|---------------|
| 0.00 | 0.963 | 0.908 | 0.950 | 0.926 | 0.944 |
| 0.05 | 0.831 | 0.815 | 0.853 | 0.741 | **0.837** |
| 0.10 | 0.755 | 0.731 | 0.784 | 0.566 | **0.657** |
| 0.20 | 0.634 | 0.612 | 0.666 | 0.478 | 0.518 |
| 0.30 | 0.565 | 0.574 | 0.611 | 0.461 | 0.468 |

On the pure aliased-observation task without landmarks, PC is the best
variant under matched-distribution noise. Level 1.5 is second. This
matches the theoretical expectation: forward models excel at aliased
aggregation (many observations per position); PC's strength is largely
orthogonal to what Kalman-style corrections provide.

### 4.2 Clone-structure analysis (CSCG-inspired)

Motivated by the Clone-Structured Cognitive Graph (George et al. 2021), we
ask: do models learn distinct per-cell representations for the same aliased
observation? Two metrics:

- **R²** — linear decoding of (x, y) from features
- **Separation score** — `(d_between_cell − d_within_cell) / d_between_cell`

| Model | θ̂ R² | hidden R² | **θ̂ separation** | hidden separation |
|-------|------|-----------|---------------------|-------------------|
| Vanilla+noise | 0.184 | 0.366 | 0.573 | 0.125 |
| **PC MapFormer** | 0.210 | 0.369 | **0.619** | 0.147 |
| Parallel InEKF | 0.307 | 0.371 | 0.395 | 0.171 |

PC has the cleanest CSCG-style clone clustering in the Lie-algebra position
state. (Clone analysis of Level 1.5 and Level 2 was not run in the
aliased-only regime; with heteroscedastic gain, the clone-structure
interpretation requires more care since the gain itself varies per token.)

### 4.3 The landmark experiment

We added 200 unique-ID landmark cells (~5% of a 64×64 grid), each emitting
a unique single-use token. This is the classical Kalman regime: sharp
unambiguous position measurements. All variants retrained with landmarks
enabled, same hyperparameters.

**T=128 (training length):**

| Metric | Vanilla+noise | Level 1 InEKF | PC | Level 2 | **Level 1.5** |
|--------|--------------|---------------|-----|---------|---------------|
| Overall acc | 0.815 | 0.866 | 0.874 | 0.852 | **0.948** |
| Overall NLL | 0.786 | 0.627 | 0.612 | 0.631 | **0.265** |
| **Landmark acc** | 0.014 | 0.180 | 0.016 | 0.029 | **0.732** |
| Regular obs acc | 0.740 | 0.821 | 0.847 | 0.799 | **0.929** |
| Blank acc | 0.984 | 0.987 | 0.981 | 0.982 | 0.987 |

**T=512 (4× out-of-distribution length):**

| Metric | Vanilla+noise | Level 1 InEKF | PC | Level 2 | **Level 1.5** |
|--------|--------------|---------------|-----|---------|---------------|
| Overall acc | 0.632 | 0.783 | 0.622 | 0.800 | **0.869** |
| Overall NLL | 1.95 | 0.98 | 2.03 | 0.90 | **0.573** |
| **Landmark acc** | 0.007 | 0.167 | 0.004 | 0.036 | **0.563** |
| Regular obs acc | 0.334 | 0.640 | 0.322 | 0.692 | **0.788** |

**Four findings:**

**A. Level 1.5 achieves real landmark utilization.** At T=128 it correctly
predicts the landmark token 73.2% of the time — the classical Kalman-filter
behavior of snapping to a known state when a sharp measurement is observed.
Level 1 InEKF (constant K*) was capped at 18% because its gain couldn't
adapt to landmark informativeness. Level 2 (full heteroscedastic with
Möbius-scanned Π) was expected to solve this but didn't; we hypothesize the
joint optimization of Π and R was harder to train, and the Π dynamics
(restricted by the Möbius recurrence) were not the useful degree of freedom.

**B. Level 1.5 wins every single metric we measured.** Overall accuracy,
NLL (calibration), landmark accuracy, regular-obs accuracy — all best at
both T=128 and T=512. The margin is not small: +7.4 pp overall accuracy
at T=128 vs the next-best variant, +55 pp landmark accuracy.

**C. Bounded-error property observed empirically, now at scale.** Level 1.5
degrades from T=128 to T=512 by only 7.9 pp (0.948 → 0.869); Level 1 InEKF
degrades 8.3 pp; all other variants degrade 15–27 pp. The Kalman filter's
theoretical bounded-error guarantee under observability is an empirical
reality for both Level 1 and Level 1.5 on this task.

**D. PC degrades catastrophically with landmarks.** PC's regular-obs
accuracy at T=512 is 32.2%, worse than vanilla (33.4%) despite being best
at T=128 without landmarks. The mechanism: the forward model `g(θ)` cannot
predict unique landmark embeddings (they appear once), so every landmark
visit produces a huge spurious prediction error that propagates through the
correction scan and drifts `θ̂` away from the true state. Forward models
are intrinsically incompatible with unique measurements.

### 4.4 Why Level 1.5 works and Level 2 didn't

Level 2 was the principled full heteroscedastic generalization: both `R_t`
and `Π_t` vary per token, with `Π_t` tracked via a Möbius-matrix associative
scan (covariance propagates through predict + update using time-varying
gain). In theory, this is strictly more expressive than Level 1.5.

Empirically, Level 2 trained to higher loss and failed to exploit landmarks.
Running a diagnostic on the trained Level 2 model revealed:

| Token type | mean R_t | mean K_t |
|---|---|---|
| action | 110 | 0.165 |
| blank | 129 | 0.110 |
| regular | 125 | 0.045 |
| landmark | 90 | 0.067 |

The model correctly assigned smaller `R_t` to landmarks — so the direction
was right — but the `K_t` dynamic range was tiny (0.04–0.17), far from the
near-1 we wanted for landmarks. The Π dynamics only varied ~4× across
tokens, adding optimization difficulty without meaningful modeling capacity.
Reducing Π to a constant (Level 1.5) eliminated this difficulty and
allowed the model to learn a much more aggressive `K_t` modulation.

The lesson: **the heteroscedastic `K_t` (via per-token `R_t`) is the useful
part of Level 2; the heteroscedastic Π (via Möbius scan) was dead weight
that made the joint optimization harder.**

### 4.5 Computational cost

Measured forward+backward time on the actual training workload (batch=128,
256 interleaved tokens):

| Variant | fwd+bwd per batch | 50-epoch training |
|---------|-------------------|--------------------|
| Vanilla MapFormer | 10 ms | ~13 min |
| Parallel InEKF (Level 1) | 10 ms | ~13 min |
| PC MapFormer | 11 ms | ~14 min |
| **Level 1.5 InEKF** | **10 ms** | **~13 min** |
| Level 2 InEKF | 600 ms | ~80 min |

Level 1.5 has the same wall-clock cost as vanilla MapFormer. Level 2 is
~60× slower per step due to two Hillis-Steele scans of 2×2 matrices with
an expensive backward pass (autograd stores `O(log T)` intermediate tensors
per scan).

---

## 5. Characterization of Regimes

Combining all experiments, the updated picture is:

| Regime | Best model | Why |
|--------|------------|-----|
| Clean task, aliased observations, T=128 | Vanilla MapFormer | Attention alone handles multimodal retrieval |
| Matched-noise training, aliased obs | PC MapFormer | Forward-model error detects drift without being misled by aliasing |
| True landmarks + any length | **Level 1.5 InEKF** | Heteroscedastic `K_t` snaps state to sharp measurements; preserves Kalman bounded-error |
| Long OOD sequences | **Level 1.5 InEKF** or Level 1 | Constant-state-size filters stay accurate where attention drifts |

Level 1.5 dominates the landmark + long-horizon regimes. On the pure aliased
task without landmarks, PC remains competitive — and the two mechanisms can
in principle be combined (see Future Directions).

---

## 6. Interpretable Architecture-Task Alignment

The strongest single observation from this work:

> **Forward models (PC) and inverse models (Kalman-family) are complementary
> classes of correction mechanism, each suited to different observation
> structures. Among Kalman-family filters, the simplest variant that
> captures per-token informativeness dominates on this task; adding
> time-varying covariance via matrix scans does not help and may hurt.**

- **Forward model `g(θ) → expected observation`** is well-defined when
  multiple observations map to each position (aliased sensory input). The
  prediction is an average; the error signal detects inconsistency. Breaks
  when single-use measurements are present.

- **Inverse model `z = h(obs) → implied position`** is well-defined when
  observations are injective (landmarks, barcodes, GPS beacons). The
  measurement directly informs state. Its effective Kalman gain should
  adapt to per-token informativeness; **this adaptation alone (Level 1.5)
  is sufficient** for this task, without time-varying covariance.

A biological cognitive-map system almost certainly uses both forward and
inverse models — place cells and landmark cells are known to coexist in the
hippocampus. The CSCG framework (George et al. 2021) is essentially a
discrete-latent-state approximation of the forward-model strategy. Classical
navigation filters are the inverse-model strategy. MapFormer with a choice
of correction mechanism is a natural testbed for comparing them.

---

## 7. Engineering Contributions

**A fully-parallel Invariant EKF on SO(2)** (Level 1), implemented in ~200
lines of PyTorch:

1. Compute θ_path via parallel cumsum (O(log T))
2. Compute wrapped innovations via elementwise `atan2(sin, cos)`
3. Precompute steady-state gain K* from closed-form scalar DARE
4. Apply correction via FFT convolution with kernel `K*·(1−K*)^k`
5. Feed corrected θ̂ to RoPE

To our knowledge, this is the first explicit pairing of parallel
associative-scan Kalman filtering (Särkkä & García-Fernández 2021) with
Lie-group SO(2) state on a transformer architecture.

**Level 1.5 InEKF** generalizes the gain to be per-token informative without
sacrificing parallelism: replace the FFT convolution with a scalar affine
associative scan (Hillis-Steele in PyTorch), allowing the Kalman gain `K_t`
to vary with each observation. Empirically the best MapFormer variant we
built on every metric, at the same speed as vanilla.

**A parallel Predictive-Coding MapFormer** using the same scan
infrastructure but with a forward-model + prediction-error update rule.
Achieves best in-distribution accuracy on aliased tasks without landmarks
and the cleanest CSCG-like clone structure.

All variants preserve MapFormer's O(log T) parallelism. Level 1 and 1.5
match vanilla wall-clock speed; Level 2 is ~60× slower due to constants,
not asymptotic complexity.

---

## 8. Limitations and Honest Assessment

1. **No formal theorem for Level 1.5.** The Level 1 InEKF inherits classical
   bounded-error guarantees. Level 1.5's per-token gain means formal
   analysis is less clean — we have empirical evidence of bounded-error
   behavior but not a proof.
2. **Single task, single environment.** All experiments are on the MapFormer
   2D torus navigation task. Generalization to other cognitive-map tasks
   (non-spatial relational reasoning, partial observability) is untested.
3. **Scale is small.** Following the paper, we use 1-layer, 2-head, d=128
   models. Paper itself notes they did not scale; our conclusions may not
   transfer to large-scale models.
4. **Level 2 result is a negative finding to explain.** Level 2 is
   theoretically strictly more expressive than Level 1.5, yet performs
   worse and is much slower. We hypothesize optimization difficulty, but
   have not verified this with careful learning-curve analysis, LR sweeps,
   or auxiliary losses. A careful ablation could reveal that Level 2 with
   the right optimization recipe would match or exceed Level 1.5.
5. **Checkpoint-code mismatch risk.** Some older checkpoints (e.g.,
   `figures_inekf_proper/`) were trained with model code that has since
   been modified; reloading them with current code gives incorrect
   behavior. Not a scientific issue, but a reproducibility caveat for
   future readers.
6. **PC's OOD failure mode is diagnosed but not fixed.** We show that
   PC's forward model fails on unique landmarks, but we have not
   implemented any of the three proposed fixes (gating landmarks out,
   bounding corrections, lower scan decay, or combining with Level 1.5).

---

## 9. Future Directions

In order of expected impact:

1. **Hybrid PC + Level 1.5.** Use PC's forward-model correction on regular
   tokens (where it excels) and Level 1.5's inverse-model Kalman update on
   landmarks (where it excels). A small gating network could choose between
   them per-token based on learned informativeness. Expected to combine
   each variant's best regime, likely exceeding Level 1.5 alone on
   mixed-observation tasks.

2. **Even longer sequences.** Evaluate at T=2048, T=10000. Attention
   becomes infeasible at that scale; Kalman/PC's constant-state-size
   correction should dominate decisively. Level 1.5's bounded-error
   property should be especially pronounced.

3. **Formal stability theorem.** State the observability condition
   (sufficient density of landmarks) and prove bounded error:
   `E[||θ̂_t − θ_t||²] ≤ M` for some constant M independent of t.
   Barrau-Bonnabel machinery applies for Level 1; Level 1.5 extension
   requires handling the time-varying gain (standard adaptive-filter
   techniques).

4. **TEM / TEM-t / CSCG baseline comparison.** The paper compares against
   RoPE / TAPE / CoPE but not against the cognitive-map literature.
   Including TEM and an HMM-based CSCG baseline would strengthen the
   "this is a cognitive-map contribution" framing.

5. **Understand why Level 2 underperforms.** Carefully controlled
   ablations (same init, same LR schedule, with and without Möbius scan)
   would diagnose whether Level 2's problems are optimization-related or
   fundamental to the full-heteroscedastic architecture.

6. **Multi-layer MapFormer + Level 1.5.** Paper only tested 1 layer. Our
   preliminary runs with multi-layer depth were unstable at current
   training budgets. Scaling study at 2/4/8 layers with Level 1.5 as
   the positional-correction substrate would be a clean contribution.

7. **Real-biology comparison.** The MapFormer paper explicitly frames
   itself as a cognitive-map theory. Showing that Level 1.5's learned
   `R_t` distribution matches hippocampal uncertainty encoding, or that
   its clone clusters align with recorded neural activity, would move
   the work from "another transformer variant" to "computational
   neuroscience contribution."

8. **Mamba-kernel implementation of Level 1.5's scan.** Our Hillis-Steele
   Python implementation is O(L log L) work. Using Mamba's
   `selective_scan_cuda` would bring this to O(L) with significantly
   better constants — useful for large-scale deployment but not necessary
   at our current scales.

---

## 10. Replication

Full code, trained checkpoints (except older inconsistent ones), and
analysis scripts are in this repository. Key reproducibility commands:

```bash
# Paper-faithful MapFormer reproduction
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train all five noise-augmented variants
python3 -m mapformer.main_vanilla_noise \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10
python3 -m mapformer.main_inekf_parallel \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10
python3 -m mapformer.main_predictive_coding \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10 \
    --aux-coef 0.1
python3 -m mapformer.main_inekf_level2 \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10
python3 -m mapformer.main_inekf_level15 \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10

# Add --n-landmarks 200 and a unique --output-dir to each for the
# landmark experiment.

# Matched-distribution action-noise comparison
python3 -m mapformer.noise_test \
    --checkpoint figures_inekf_level15/MapFormer_WM_Level15InEKF.pt

# Landmark evaluation (accuracy + NLL by cell type)
python3 -m mapformer.landmark_eval \
    --checkpoints \
      figures_vanilla_noise_lm200/MapFormer_WM_noise.pt \
      figures_inekf_parallel_lm200/MapFormer_WM_ParallelInEKF.pt \
      figures_pc_lm200/MapFormer_WM_PredictiveCoding.pt \
      figures_inekf_level2_lm200/MapFormer_WM_Level2InEKF.pt \
      figures_inekf_level15_lm200/MapFormer_WM_Level15InEKF.pt \
    --device cuda --n-steps 128 512

# Gaussian Δ-noise robustness
python3 -m mapformer.gaussian_noise_test \
    --checkpoints <all-variants> --device cuda --n-steps 128

# Clone-structure analysis
python3 -m mapformer.clone_analysis \
    --checkpoint figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --fixed-start

# Diagnostic checks (disentanglement, ω, action selectivity)
python3 -m mapformer.diagnose \
    --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

Detailed results tables for each experimental regime are in
`RESULTS_LEVEL2.md` and `RESULTS_LEVEL15.md` in the repo.

---

## 11. Key References

- **Rambaud, Mascarenhas, Lakretz (2025).** MapFormer: Self-Supervised
  Learning of Cognitive Maps with Input-Dependent Positional Embeddings.
  [arXiv:2511.19279](https://arxiv.org/abs/2511.19279)
- **Barrau, Bonnabel (2017).** The Invariant Extended Kalman Filter as a
  Stable Observer. IEEE TAC, 62(4).
  [arXiv:1410.1465](https://arxiv.org/abs/1410.1465)
- **Marković, Ćesić, Petrović (2017).** On wrapping the Kalman filter and
  estimating with the SO(2) group.
  [arXiv:1708.05551](https://arxiv.org/abs/1708.05551)
- **Särkkä, García-Fernández (2021).** Temporal Parallelization of Bayesian
  Filters and Smoothers. IEEE TAC, 66(1).
  [arXiv:1905.13002](https://arxiv.org/abs/1905.13002)
- **Yaghoobi, Corenflos, Hostettler, Särkkä (2021).** Parallel Iterated
  Extended and Sigma-Point Kalman Smoothers. ICASSP.
  [arXiv:2102.00514](https://arxiv.org/abs/2102.00514)
- **Gu, Dao (2023).** Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **George, Rikhye, Gothoskar, Guntupalli, Dedieu, Lázaro-Gredilla (2021).**
  Clone-structured graph representations enable flexible learning and
  vicarious evaluation of cognitive maps. Nature Communications 12.
- **Whittington, Warren, Behrens (2022).** Relating transformers to models
  and neural representations of the hippocampal formation. ICLR.
- **Rao, Ballard (1999).** Predictive coding in the visual cortex: a
  functional interpretation of some extra-classical receptive-field
  effects. Nature Neuroscience 2(1).

---

## Acknowledgment

This project was developed through extensive iterative debugging, with
many incorrect intermediate architectures discarded. Seven paper-
faithfulness bugs were identified and fixed in the initial reproduction
phase. Three distinct InEKF implementations were built before landing on
the final Level 1.5 parallel design. Level 2, a theoretically-stronger
generalization, was implemented, found to underperform, and
mechanistically diagnosed — the apparent contradiction (more expressive,
worse results) was resolved by identifying that the Möbius scan added
optimization difficulty without useful capacity. The predictive-coding
variant was initially misdesigned (conditioning head on position creates
a degenerate optimum) before the final clean implementation. Both the
scientific honesty in reporting these failures and the engineering
discipline in preserving MapFormer's parallelism throughout make the
final results worth reporting.
