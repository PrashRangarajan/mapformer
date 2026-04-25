# Hippocampal / Grid-Cell Correspondence Analysis

Three tests connecting trained MapFormer representations to known properties of place / grid / boundary cells in entorhinal cortex and hippocampus.

**TL;DR — partially confirming, partially falsifying.** The basic informativeness signal (blank vs non-blank) IS encoded by the R_t head, but the predicted "landmark < aliased" ordering does **not** appear in our trained models. Path-integrator blocks are *not* hexagonal grid-cell-like in the Sargolini sense — they're 1D phase-clocks, not 2D periodic fields. Honest negative result that informs paper framing.

## Test A — Spatial rate maps

Each path-integrator block is shown as a heatmap of cos(θ̂) by (x, y), averaged over a fresh-environment trajectory. Real grid cells (entorhinal) produce hexagonal periodic patterns; real place cells (hippocampal CA1) produce single-peak fields.

![Rate maps](paper_figures/fig7_rate_maps.png)

**Grid score (Sargolini et al. 2006 — hexagonal autocorrelation): higher = more grid-like. > 0.3 typically considered grid-like.**

| Variant | mean across blocks | max across blocks |
|---|---|---|
| Vanilla | −0.034 | +0.145 |
| VanillaEM | −0.030 | +0.258 |
| Level1 | −0.069 | +0.088 |
| Level15 | −0.043 | +0.222 |
| Level15EM | −0.041 | +0.124 |

**Result: no variant reaches grid-cell-like scores.** Best max is VanillaEM at 0.258 — below the conventional 0.3 threshold. Means are around zero.

**Why this isn't TEM-style grid-cell emergence.** The path integrator imposes a *fixed* `cumsum(ω·Δ) + cos/sin` structure rather than learning 2D position fields from scratch. Each block produces a 1D phase clock — a periodic signal whose phase encodes one component of position. Two such clocks at incommensurate frequencies can encode 2D position, but the *individual block activations* trace stripe-like patterns in (x,y), not hexagonal ones. To get true hexagonal grid cells, you'd need either: (a) a 2D rotation in SO(2)×SO(2) per block with a 60° phase offset, or (b) learned (not fixed) position embeddings that the network is free to shape. TEM does (b); MapFormer does neither.

This is **not a flaw** — MapFormer's structure is more constrained but easier to train and analyse. It just means we shouldn't claim grid-cell correspondence; what MapFormer learns is **the underlying torus topology** rather than the *specific cellular code* used by entorhinal cortex.

## Test B — Per-token measurement noise R_t at landmarks

**Hypothesis (theoretical):** R_t should be small at *unique* tokens (landmarks → sharp posterior, high informativeness for the filter), large at *blank* tokens (no position information), and intermediate at *aliased* obs. This mirrors entorhinal "boundary cells" (Solstad et al. 2008) and "object cells" (Lever et al. 2009) firing selectively at distinctive features.

![R at landmarks](paper_figures/fig8_R_landmark.png)

| Variant | landmark ⟨log R⟩ | aliased obs ⟨log R⟩ | blank ⟨log R⟩ |
|---|---|---|---|
| Level15 | +0.624 | **+0.385** (lowest) | +0.834 (highest) |
| Level15EM | +3.837 | +4.464 (highest) | **+1.780** (lowest) |

**Result: predicted ordering NOT observed in either variant.**

- **Level15-WM** ordering: **aliased < landmark < blank**. Aliased obs (0.385) get the *lowest* R, not landmarks.
- **Level15-EM** ordering: **blank < landmark < aliased**. Blank tokens get the *lowest* R.

Neither matches the predicted `landmark < aliased < blank`. **The blank-vs-non-blank distinction *is* learned in both cases** (∆ ≈ 0.4 in WM, ∆ ≈ 2.5 in EM), but the within-non-blank ordering is data-driven rather than theoretical.

**Interpretation.** R_t is end-to-end trained on revisit prediction loss, not on a Bayesian-optimal "informativeness" objective. The optimal R for the *training task* depends on:

1. **Frequency of tokens.** Aliased obs occur on ~50% of cells, landmarks on ~5%. The R_t head sees more training signal for aliased.
2. **Predictability under path integration alone.** Aliased obs are ~257-way ambiguous, but for cells the model has visited before (the prediction target), the answer is unique given good path integration. So z_t ≈ θ_path_t, innovation is small, and Kalman correction is moot — but the head has learned that aliased *measurements* are "trustworthy" because the model can reliably predict their target.
3. **Backbone interaction.** WM treats correction as additive perturbation to a single rotation; EM treats it as input to the position branch of a Hadamard product. The two interact differently with R, so the optimal R for EM is shifted upward (R≈20–87) relative to WM (R≈1.4–2.3).

**Honest conclusion.** The "boundary-cell correspondence" claim is **not supported** by our trained Level 1.5 models. R_t encodes informativeness, but not in the way pure Kalman theory predicts. This is a real result — Level 1.5's R_t head is a learned task-specific structure, not a Bayesian-optimal one.

## Test C — ω frequency spectrum (grid-cell modules)

Stensola et al. (2012, *Nature*) showed grid-cell modules in entorhinal cortex have spacings following a roughly **√2 geometric ratio** — a discrete log-uniform structure across modules. MapFormer's geometric initialisation gives a similar log-uniform structure; we plot trained ω to see whether training preserves it.

![ω modules](paper_figures/fig9_omega_modules.png)

Solid lines: trained ω per variant (sorted high → low). Dashed: untrained geometric init for reference.

**Result.** Trained ω across all variants approximately preserves the geometric log-uniform structure of the init. The trained curves are close to the dashed init line, with mild deviations — training perturbs ω but doesn't dramatically reshape the spectrum. **This is the one quantitative correspondence to neural data that holds**: MapFormer's frequency spectrum, like grid-cell module spacings, is approximately geometric across scales.

But the correspondence is largely **inherited from the init**, not earned by training — so it's a softer claim than "the model discovered grid-cell-like organisation."

## Overall summary

| Test | Predicted | Observed | Verdict |
|---|---|---|---|
| **A. Hexagonal grid cells** | Hexagonal periodic rate maps | Stripe-like 1D phase clocks | ❌ falsified — architectural artifact, not learned grid cells |
| **B. R_t at landmarks** | landmark < aliased < blank | aliased < landmark < blank (WM); blank < landmark < aliased (EM) | ❌ falsified — basic blank/non-blank distinction holds, fine-grained ordering does not |
| **C. ω modules** | Geometric (√2) spacing | Approximately geometric (largely from init) | ✓ holds, but trivially |

**What this means for the paper:**

1. Don't claim MapFormer's representations match entorhinal grid cells in the strong Sargolini-grid-score sense. They don't.
2. Don't claim Level 1.5's R_t matches boundary/object cell firing patterns. The basic informativeness gradient is there, but the specific ordering isn't.
3. Do claim: the **frequency-spectrum structure** is broadly compatible with grid-cell modular organisation (Stensola √2), and **the model learns to distinguish informative from non-informative tokens** (just not in a Bayesian-optimal way).

This is a **net-honest** outcome: the paper's main claims (Level 1.5 wins on noise/landmarks/calibration, Mamba can't do this task) are unaffected. The neuroscientific correspondence we hoped to add as a bonus is more nuanced than predicted. Future work could explore: training with a Bayesian-informativeness regulariser to align R_t with theory, or learning 2D position fields end-to-end (TEM-style) to recover hexagonal grid structure.

---
*Auto-generated by `hippocampal_analysis.py`. Generated 2026-04-24.*
