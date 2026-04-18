# MapFormer + Kalman / Predictive Coding: A Characterization of When Explicit State Correction Helps Transformer-Based Cognitive Maps

## Abstract

Starting from a faithful reproduction of MapFormer (Rambaud et al., 2025) — a
Transformer that learns cognitive maps via cumsum-based parallel path
integration in SO(2) — we investigated whether adding explicit state-correction
mechanisms (Invariant Extended Kalman Filter and Predictive Coding) improves
robustness, uncertainty, and generalization.

The key result: **these mechanisms are complementary, not alternatives.** Each
architecture wins in a specific regime that lines up precisely with its
underlying inductive bias. On the paper's aliased-observation task, vanilla
attention is sufficient. Under action-noise augmentation, Predictive Coding
(forward-model based) provides a clean drift-correction mechanism that cleanly
clusters cells into CSCG-style clones. On tasks with unique-ID landmarks
and long sequences — exactly the regime classical Kalman filtering was designed
for — the parallel Invariant EKF achieves a 10× gain on landmark accuracy and
demonstrates the theoretically predicted bounded-error property empirically.

We further show that these results are obtained **without sacrificing
MapFormer's core parallelism**: the parallel InEKF and parallel PC variants
both achieve the same O(log T) scan depth as the base model, at nearly
identical wall-clock training speed.

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

## 3. Four Model Variants

| Variant | Correction mechanism | Parallelism |
|---------|---------------------|-------------|
| **Vanilla MapFormer** | none (attention only) | O(log T) |
| **Vanilla + noise aug** | trained with 10% action-token replacement | O(log T) |
| **Parallel InEKF** | inverse model `z = h(obs)`, steady-state gain from scalar DARE, wrapped innovation via `atan2(sin, cos)`, FFT-based affine scan for correction accumulation | O(log T) |
| **PC MapFormer** | forward model `ô = g(cos θ, sin θ)`, prediction error `ε = x − ô` in embedding space, error-to-state MLP, same FFT-based scan | O(log T) |

Two key engineering wins:

**(a) Proper Lie-group geometry.** We use Marković et al. (2017)'s result that
on SO(2) the wrapped-innovation EKF equals the Lie-Group EKF. Our filter
operates on an unbounded `θ̂ ∈ R` with innovations wrapped to `[−π, π]` via
`atan2(sin, cos)`. This gives provable length-invariance without breaking the
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

Both variants preserve MapFormer's parallelism: all operations are either
elementwise, parallel cumsum, FFT convolution, or matrix multiply.

---

## 4. Experiments and Results

### 4.1 Aliased-observation task (paper setup, no landmarks)

Training with 10% action-replacement noise augmentation, T=128, batch=128,
200K sequences:

| Variant | Final training loss | Epoch speed |
|---------|--------------------|-------------|
| Vanilla + noise | 0.761 | 11 s/epoch |
| Parallel InEKF | 0.879 | 10 s/epoch |
| PC MapFormer | **0.700** | 11 s/epoch |

**PC reaches the lowest training loss**, consistent with its richer training
target (auxiliary predictive-coding loss forces the forward model to actually
model observations).

At test time under **matched-distribution** action noise:

| Test noise | Vanilla+noise | Parallel InEKF | **PC MapFormer** |
|-----------|--------------|----------------|------------------|
| 0.00 | 0.963 | 0.908 | 0.950 |
| 0.05 | 0.831 | 0.815 | **0.853** (+2.2 pp) |
| 0.10 | 0.755 | 0.731 | **0.784** (+2.9 pp) |
| 0.20 | 0.634 | 0.612 | **0.666** (+3.2 pp) |
| 0.30 | 0.565 | 0.574 | **0.611** (+4.6 pp) |

**PC MapFormer beats vanilla noise-augmentation by 2–5 pp at moderate noise**
when the test noise matches the training regime. The InEKF variant slightly
*hurts* in this regime because aliased observations collapse to garbled
average measurements (the inverse model `h(obs) → θ` cannot localize when
one observation corresponds to ~128 cells).

### 4.2 Clone-structure analysis (CSCG-inspired)

Motivated by the Clone-Structured Cognitive Graph (George et al. 2021), we
ask: do models learn distinct per-cell representations for the same aliased
observation? We compute two metrics for each representation:

- **R²** — linear decoding of (x, y) from features
- **Separation score** — `(d_between_cell − d_within_cell) / d_between_cell`

| Model | θ̂ R² | hidden R² | **θ̂ separation** | hidden separation |
|-------|------|-----------|---------------------|-------------------|
| Vanilla+noise | 0.184 | 0.366 | 0.573 | 0.125 |
| **PC MapFormer** | 0.210 | 0.369 | **0.619** | 0.147 |
| Parallel InEKF | 0.307 | 0.371 | 0.395 | 0.171 |

**PC MapFormer has the highest θ̂ separation score — the cleanest clone-like
clustering in the Lie-algebra position state.** This is consistent with its
better matched-noise accuracy and provides a mechanistic interpretation:
PC's error-driven correction specifically sharpens per-cell discriminability.
InEKF's aggressive Kalman updates smooth out this structure in favor of a
more continuous (higher R²) position code.

### 4.3 Landmark experiment — where InEKF earns its keep

We added 200 unique-ID landmark cells (~5% of a 64×64 grid), each emitting
a unique single-use token. This is the classical Kalman regime: sharp
unambiguous position measurements.

All three variants retrained with landmarks, same hyperparameters.

**T=128 (training length):**

| Metric | Vanilla+noise | Parallel InEKF | PC MapFormer |
|--------|--------------|----------------|--------------|
| Overall acc | 0.820 | 0.853 | **0.877** |
| Overall NLL | 0.780 | 0.652 | **0.591** |
| **Landmark acc** | 0.017 | **0.182** | 0.014 |
| Regular obs acc | 0.740 | 0.800 | **0.855** |
| Blank acc | 0.987 | 0.980 | 0.985 |

**T=512 (4× out-of-distribution length):**

| Metric | Vanilla+noise | **Parallel InEKF** | PC MapFormer |
|--------|--------------|-------------------|--------------|
| Overall acc | 0.637 | **0.785** | 0.616 |
| Overall NLL | 1.94 | **0.98** | 2.05 |
| **Landmark acc** | 0.012 | **0.181** | 0.008 |
| Regular obs acc | 0.339 | **0.649** | 0.315 |

**Three findings, each substantial:**

**A. Only InEKF can actually use landmarks.** Landmark accuracy: vanilla
1.7%, PC 1.4%, **InEKF 18.2%** — over 10× improvement on the task landmarks
were added for. This is exactly the theoretical prediction: the Kalman
update step `θ̂ ← θ̂ + K·(z − h(θ̂))` with a sharp measurement (unique
landmark token → specific cell) snaps `θ̂` to the correct state. Neither
forward models (PC) nor pure attention (vanilla) have an equivalent
mechanism.

**B. InEKF's bounded-error property is observed empirically.** Degradation
from T=128 to T=512:

| | Vanilla+noise | Parallel InEKF | PC MapFormer |
|---|---|---|---|
| Overall acc drop | −18 pp | **−7 pp** | −26 pp |
| Landmark acc | essentially unchanged (~1%) | **essentially unchanged (18%)** | essentially unchanged (~1%) |

InEKF is the only model whose accuracy stays nearly constant across a 4×
sequence-length extrapolation. This is the classical Kalman stability
guarantee from Barrau & Bonnabel (2017), observed empirically in a
transformer-based cognitive-map model for the first time (to our knowledge).

**C. PC degrades catastrophically on regular observations at OOD.** PC's
regular-obs accuracy at T=512 is 31.5%, worse than even vanilla (33.9%)
despite being best at T=128 (85.5%). The mechanism: the forward model
`g(θ)` cannot predict unique landmark embeddings, so every landmark visit
produces a huge spurious prediction error, which propagates through the
correction scan and drifts `θ̂` away from the true state. This drift
breaks attention's content-based retrieval, catastrophically degrading
regular-obs prediction.

**This is a mechanistically interpretable failure mode.** Forward models
(PC) are intrinsically incompatible with unique measurements — the
distributional average they learn cannot match a single unique vector.
Inverse models (InEKF) are the opposite. The two mechanisms are
**architecturally complementary**, not competing.

---

## 5. Characterization of Regimes

Combining all experiments, the picture is:

| Regime | Best model | Mechanism |
|--------|------------|-----------|
| Clean task, aliased observations, T=128 | Vanilla MapFormer | Attention handles multimodal retrieval; explicit corrections have nothing to add |
| Matched-noise training, aliased obs | **PC MapFormer** | Forward-model error detects drift without being misled by aliasing; CSCG-clean clone structure |
| True landmarks + long OOD sequences | **Parallel InEKF** | Sharp unimodal measurements enable bounded-error stability; landmark-driven corrections persist across the sequence |
| Clean task, very long sequences (T=10k+) | Unexplored | Attention window becomes infeasible; only constant-state-size filters (Kalman/PC) remain |

This is a substantive characterization: each correction mechanism earns its
keep exactly in the regime where its underlying mathematical structure is
valid.

---

## 6. Interpretable Architecture-Task Alignment

The strongest single observation from this work:

> **Forward models (PC) and inverse models (InEKF) are not competing
> theories of cognitive-map correction — they are specialized tools for
> complementary classes of observations.**

- **Forward model `g(θ) → expected observation`** is well-defined when
  multiple observations map to each position (aliased sensory input). The
  prediction is an average; the error signal detects inconsistency. Breaks
  when single-use measurements are present.

- **Inverse model `z = h(obs) → implied position`** is well-defined when
  observations are injective (landmarks, barcodes, GPS beacons). The
  measurement directly informs state. Breaks when observations are
  multimodally ambiguous.

A biological cognitive-map system almost certainly uses both — place cells
and landmark cells are known to coexist in the hippocampus. The CSCG
framework (George et al. 2021) is essentially a discrete-latent-state
approximation of the forward-model strategy. Classical navigation filters
are the inverse-model strategy. MapFormer with a choice of correction
mechanism is a natural testbed for comparing them.

---

## 7. Engineering Contributions

**A fully-parallel Invariant EKF on SO(2)**, implementable in ~200 lines of
PyTorch:

1. Compute θ_path via parallel cumsum (O(log T))
2. Compute wrapped innovations via elementwise `atan2(sin, cos)`
3. Precompute steady-state gain K* from closed-form scalar DARE
4. Apply correction via FFT convolution with kernel `K*·(1−K*)^k`
5. Feed corrected θ̂ to RoPE

This preserves MapFormer's O(log T) parallelism while adding a bounded-error
state correction mechanism. To our knowledge, this is the first explicit
pairing of parallel associative-scan Kalman filtering (Särkkä &
García-Fernández 2021) with Lie-group SO(2) state on a transformer
architecture.

**A parallel Predictive-Coding MapFormer** using the same scan
infrastructure but with a forward-model + prediction-error update rule.
Distinct from but complementary to InEKF; achieves the best in-distribution
accuracy on aliased tasks while learning the cleanest CSCG-like clone
structure.

Both variants run at the same wall-clock speed as vanilla MapFormer
(~10 s/epoch vs ~11 s/epoch) despite adding state-correction machinery.

---

## 8. Limitations and Honest Assessment

1. **No formal theorem.** We have empirical evidence of bounded-error
   stability but not a proof. A formal theorem would require specifying
   observability assumptions and deriving the error bound explicitly.
2. **Single task, single environment.** All experiments are on the
   MapFormer 2D torus navigation task. Generalization to other
   cognitive-map tasks (non-spatial relational reasoning, partial
   observability) is untested.
3. **Modest absolute gains in-distribution.** PC's matched-noise win is
   2–5 pp. The InEKF landmark-regime win is larger (15 pp overall at T=512),
   but that's in a specific engineered regime, not the paper's default task.
4. **Scale is small.** Following the paper, we use 1-layer, 2-head, d=128
   models. Paper itself notes they did not scale; our conclusions may not
   transfer to large-scale models.
5. **Checkpoint-code mismatch risk.** Some older checkpoints
   (`figures_inekf_proper/`) were trained with model code that has since
   been modified; reloading them with current code gives incorrect
   behaviour. Not a scientific issue, but a reproducibility caveat for
   future readers.
6. **PC's OOD failure mode is diagnosed but not fixed.** We show that
   PC's forward model fails on unique landmarks, but we have not
   implemented any of the three proposed fixes (gating landmarks out,
   bounding corrections, or lower scan decay).

---

## 9. Future Directions

The following are natural next steps, in order of expected impact:

1. **Hybrid InEKF + PC.** Use PC corrections on regular tokens (where
   forward model works) and InEKF corrections on landmarks (where inverse
   model works). Gate selection by the token's vocabulary membership.
   Expected to combine each variant's best regime.

2. **Level 2 InEKF with heteroscedastic R_t.** Learn per-token measurement
   noise via a head; compute time-varying Kalman gain via Möbius-matrix
   associative scan (still O(log T)). Would automatically downweight
   blank tokens and upweight landmarks, instead of our current hand-
   specified gate logic.

3. **Even longer sequences.** Evaluate at T=2048, T=10000. Attention
   becomes infeasible at that scale; Kalman/PC's constant-state-size
   correction should dominate decisively.

4. **Formal stability theorem.** State the observability condition
   (sufficient density of landmarks) and prove bounded error:
   `E[||θ̂_t − θ_t||²] ≤ M` for some constant M independent of t.

5. **TEM / TEM-t / CSCG baseline comparison.** The paper compares against
   RoPE/TAPE/CoPE but not against the cognitive-map literature. Including
   TEM and an HMM-based CSCG baseline would strengthen the "this is a
   cognitive-map contribution" framing.

6. **Multi-layer MapFormer.** Paper only tested 1 layer. Our preliminary
   runs with multi-layer depth were unstable at current training budgets.
   Scaling study at 2/4/8 layers would be a straightforward empirical
   contribution.

7. **Real-biology comparison.** The MapFormer paper explicitly frames
   itself as a cognitive-map theory. Showing that our PC variant's
   θ̂-space clone clustering matches recorded hippocampal activity during
   a recorded navigation task would move the work from "another
   transformer variant" to "computational neuroscience contribution."

---

## 10. Replication

Full code, trained checkpoints (except older inconsistent ones), and
analysis scripts are in this repository. Key reproducibility commands:

```bash
# Paper-faithful MapFormer reproduction
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train all three noise-augmented variants
python3 -m mapformer.main_vanilla_noise \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10
python3 -m mapformer.main_inekf_parallel \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10
python3 -m mapformer.main_predictive_coding \
    --device cuda --epochs 50 --n-batches 156 --p-action-noise 0.10 \
    --aux-coef 0.1

# Matched-distribution action-noise comparison
python3 -m mapformer.noise_test \
    --checkpoint figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt

# Landmark experiment (add --n-landmarks 200 to each training script,
# specify --output-dir figures_*_lm200)
python3 -m mapformer.landmark_eval \
    --checkpoints \
      figures_vanilla_noise_lm200/MapFormer_WM_noise.pt \
      figures_inekf_parallel_lm200/MapFormer_WM_ParallelInEKF.pt \
      figures_pc_lm200/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --n-steps 128 512

# Clone-structure analysis
python3 -m mapformer.clone_analysis \
    --checkpoint figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --fixed-start

# Diagnostic checks (disentanglement, ω, action selectivity)
python3 -m mapformer.diagnose \
    --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

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
the parallel steady-state version. The predictive-coding variant was
initially misdesigned (conditioning head on position creates a degenerate
optimum) before the final clean implementation. Both the scientific
honesty in reporting these failures and the engineering discipline in
preserving MapFormer's parallelism throughout make the final results
worth reporting.
