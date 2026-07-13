# Session — Hierarchical MapFormer extensions + Axis 3 Kalman cascade

Handoff notes so a fresh chat can pick up the work. Written from a
discussion about neuroscience-motivated hierarchical extensions and the
first-pass implementation of Axis 3 (Kalman cascade) as
`Level15Cascade`.

## Context / starting question

Brain memory is hierarchical along several axes (temporal, spatial,
episodic-vs-semantic). MapFormer currently is not, in any explicit way.
Question: can MapFormer be recast as a hierarchical model, and which
axis is most tractable + most neuroscientifically defensible?

## Five candidate axes (with debate)

### Axis 1 — Spatial-frequency module hierarchy
- **Source:** Stensola et al. 2012 (√2 grid module spacing in MEC),
  Doeller/Barry/Burgess 2010
- **Where in MapFormer:** the ω spectrum is *already* this — geometric
  log-uniform frequencies across blocks (see [model.py:88](model.py:88))
- **Pros:** cheap to make explicit (block grouping), direct neural
  correspondence, preserves parallelism
- **Cons:** [HIPPOCAMPAL_ANALYSIS.md](HIPPOCAMPAL_ANALYSIS.md) already
  showed individual blocks are stripe-like 1D phase clocks, not
  hexagonal grid cells. Cross-scale gating mechanism is
  architecturally underconstrained.
- **Neuro-plausibility:** ★★★★☆ (spectrum matches; cell-level
  organisation does not)

### Axis 2 — Temporal hierarchy / retrieval buffer
- **Source:** Atkinson-Shiffrin multi-store, Complementary Learning
  Systems (McClelland et al. 1995), Hasson et al. 2015 hierarchical
  temporal receptive fields
- **Extension:** persistent buffer of compressed past-sequence
  summaries; write policy on completion, read via retrieval at
  inference
- **Pros:** deepest mapping to cognitive-neuroscience canon; solves
  MapFormer's real limitation that sequences are currently
  independent; enables true cross-environment transfer
- **Cons:** biggest engineering lift by far; risks becoming a generic
  memory-augmented transformer that loses MapFormer's specific
  Lie-group structure; biology is descriptive not prescriptive
- **Neuro-plausibility:** ★★★★☆ conceptually, ★★☆☆☆ mechanistically

### Axis 3 — Kalman cascade (multi-timescale filters)
- **Source:** Bastos et al. 2012 canonical microcircuit; Rao-Ballard
  1999 hierarchical predictive coding; Kiebel et al. 2008
- **Extension:** fast per-token Kalman + slow per-chunk Kalman
  operating on residual innovations
- **Pros:** mathematically clean; testable in existing benchmark
  (paper's central length-generalisation axis); novel — no cascaded
  Kalman transformer in the literature; preserves parallelism
- **Cons:** the "cascade" formalism is engineering-inspired more than
  biology-inspired; chunk length is an unprincipled hyperparameter;
  empirical gain uncertain
- **Neuro-plausibility:** ★★★☆☆ (idea well-supported, our formalism
  is a simplification)

### Axis 4 — Depth hierarchy (per-layer InEKF corrections)
- **Source:** Felleman & Van Essen 1991 cortical hierarchy; Rao-Ballard
  1999
- **Extension:** multi-layer MapFormer where each layer applies its
  own Kalman correction using layer-local features
- **Pros:** direct implementation of Rao-Ballard predictive coding;
  no new infrastructure; would probe *depth* in MapFormer for the
  first time
- **Cons:** CLAUDE.md flags preliminary multi-layer runs as unstable;
  attention already shares information across layers; predictive
  coding needs bidirectional message passing to be authentic
- **Neuro-plausibility:** ★★★★☆

### Axis 5 — Episodic/semantic split (TEM-style)
- **Source:** grid cells (Hafting 2005) + place cells (O'Keefe 1971);
  formalised by TEM (Whittington 2020, 2022)
- **Extension:** InEKF (semantic/structural) + Hopfield content buffer
  (episodic/content)
- **Pros:** **highest neuro-plausibility of any option** — direct
  hippocampal-entorhinal anatomy match; TEM predicts empirical
  remapping patterns quantitatively
- **Cons:** `TEMFaithful` already exists in repo and matches Level 1.5
  on lm200 (see [MULTISEED_FOLLOWUP.md](MULTISEED_FOLLOWUP.md)) — so
  "add TEM to MapFormer" risks marginal contribution; combining is
  architecturally awkward (Kalman in algebra, Hopfield in features);
  Hopfield writes may break O(log T) scan
- **Neuro-plausibility:** ★★★★★

### Comparison table

| Axis | Neuro-plaus. | Effort | Tractability | Novelty | Paper fit |
|---|---|---|---|---|---|
| 1. Spatial modules | ★★★★☆ | Low | High | Low | Good |
| 2. Retrieval buffer | ★★☆☆☆ | Very high | Low | High | Poor |
| 3. Kalman cascade | ★★★☆☆ | Medium | **High** | **High** | Excellent |
| 4. Depth hierarchy | ★★★★☆ | Medium* | Medium | Medium | Good |
| 5. Ep/sem split | ★★★★★ | High | Medium | Low† | Excellent |

*After multi-layer stability work. †TEMFaithful already implements this.

**Two honest tensions:**
1. Neuro-plausibility vs novelty — most neuro-plausible options
   (Axis 5) are already implemented by cognitive-neuro groups; most
   novel (Axis 3) is less directly grounded in biology.
2. Making MapFormer better vs cognitive-neuroscience-legible — Axes
   1/3/4 improve what MapFormer already does; Axes 2/5 extend it
   into new territory. Different papers.

**Recommendation reached in session:** Axis 3 as the concrete
implementation, Axis 1 mentionable as "already implicitly present in
the ω spectrum" alongside a Stensola citation. Axis 3 chosen because:
tractable in the current codebase, testable in the paper's central
benchmark (length generalisation at long T), preserves parallelism,
and if it works gives a genuinely novel result.

## Axis 3 — design decisions

Discussed and settled before implementation:

1. **Two-level cascade** (not K-level recursive). Minimum viable
   version; extend later if the two-level result is promising.
2. **Slow filter corrects state** (not gain). `θ̂ = θ_path + d_fast +
   d_slow`. Maps cleanly to hierarchical predictive coding (each
   level produces its own state estimate).
3. **Slow filter measures residual innovations from the fast filter**
   (not raw pooled measurements). This is the true "cascade" — the
   slow filter only corrects what the fast filter failed to explain.
   An optimal fast filter would leave zero-mean white residuals; a
   *misspecified* fast filter (which Level 1.5 arguably is, since Π
   is a learned scalar rather than the true posterior) leaves
   structured residuals the slow filter can exploit.
4. **Standalone module** `InEKFCascade`, first paired with Level 1.5
   as `Level15Cascade`. In principle composable with any InEKF
   backbone.
5. **Chunk length as constructor arg**, default 32. Not exposed
   through train_variant.py's CLI yet — can add later if a sweep is
   worth running.
6. **Slow filter initialised near a no-op** — `log_R_head_slow` bias
   set to +3.0 so K_slow ≈ 0.05 at start. Model behaves like Level
   1.5 initially; slow filter only grows if there's exploitable
   residual structure. Prevents early-training destabilisation from
   random slow corrections.

## What was implemented (commit `0e3ea19`)

- **[model_inekf_cascade.py](model_inekf_cascade.py)** — new module.
  `InEKFCascade` implements the fast + slow filter,
  `MapFormerWM_Level15Cascade` wraps it into a full MapFormer-WM.
- **[train_variant.py](train_variant.py)** — registered
  `"Level15Cascade"` in `VARIANT_MAP`.

The forward pass (schematic):

```
# Fast filter (Level 1.5 mechanism)
log_R_fast = MLP_R(x_t)      → R_fast_t
K_fast_t   = Π_fast / (Π_fast + R_fast_t)
d_fast     = scan(K_fast, ν_fast)
θ̂_fast    = θ_path + d_fast

# Slow filter on chunk-pooled residuals
ν_resid_t  = wrap(z_t − θ̂_fast_t)
ν̄_c        = mean(ν_resid over chunk c)
content̄_c  = mean(x over chunk c)
log_R_slow = MLP_slow(content̄_c)   → R_slow_c
K_slow_c   = Π_slow / (Π_slow + R_slow_c)
D_slow_c   = scan(K_slow, ν̄)           # over chunk endpoints, length L/C
d_slow_t   = D_slow[chunk(t)]          # piecewise-constant broadcast

# Combine
θ̂_t       = θ_path_t + d_fast_t + d_slow_t
```

Both scans use the existing `assoc_scan_affine_scalar` from
[model_inekf_level15.py:41](model_inekf_level15.py:41). Overall depth
stays O(log L).

Diagnostics saved on the model for post-hoc analysis:
`last_d_fast, last_d_slow, last_K_fast, last_K_slow, last_R_slow`.

## Training commands

```bash
# Clean task
for seed in 0 1 2; do
    python3 -m mapformer.train_variant \
        --variant Level15Cascade --seed $seed \
        --epochs 50 --n-batches 156 \
        --output-dir runs/Level15Cascade_clean/seed$seed
done

# Noise task
for seed in 0 1 2; do
    python3 -m mapformer.train_variant \
        --variant Level15Cascade --seed $seed \
        --epochs 50 --n-batches 156 --p-action-noise 0.10 \
        --output-dir runs/Level15Cascade_noise/seed$seed
done

# Landmark task (where correction pays off most)
for seed in 0 1 2; do
    python3 -m mapformer.train_variant \
        --variant Level15Cascade --seed $seed \
        --epochs 50 --n-batches 156 --n-landmarks 200 \
        --output-dir runs/Level15Cascade_lm200/seed$seed
done
```

## Not yet done — eval-side registration

Eval scripts have their own local `VARIANT_CLS` dicts. To evaluate
`Level15Cascade` post-training, add the import + registration to:

- [long_sequence_eval.py](long_sequence_eval.py)
- [zero_shot_eval.py](zero_shot_eval.py)
- [held_out_obs_eval.py](held_out_obs_eval.py)
- [orchestrator_finalize.sh](orchestrator_finalize.sh) (if using the
  paper-tables pipeline)

Diff pattern (one line + one registry entry per file):

```python
from mapformer.model_inekf_cascade import MapFormerWM_Level15Cascade
# ...
VARIANT_CLS = {
    # ...
    "Level15Cascade": MapFormerWM_Level15Cascade,
    # ...
}
```

## What to look for when results come back

The cascade hypothesis lives or dies on whether the slow filter
learns to do anything. Three diagnostics:

1. **`last_K_slow` distribution over training.** Initialised at
   ~0.05. If it stays near 0.05, the slow filter never found
   exploitable structure — cascade is architectural cruft and Level
   1.5 alone is doing all the work. If it grows to 0.2+ (especially
   selectively, per chunk or token type), the cascade is actually
   contributing.
2. **`||d_slow|| / ||d_fast||` magnitude ratio.** If < 0.05, slow
   correction is negligible. If 0.1–0.5, slow correction is a
   meaningful chunk-of-θ shift.
3. **Length-generalisation slope.** Strongest test of the multi-
   timescale claim: cascade should beat Level 1.5 *by more* at
   T=2048 than at T=512. Slow corrections compound more helpfully
   as sequences get longer.

## Failure modes to expect

- **Cascade collapses to Level 1.5** — K_slow → 0 during training.
  Honest negative result: says Level 1.5's implicit prior variance
  Π is already well-calibrated. In that case, worth reporting as a
  "we thought this would help; it didn't for this reason."
- **Helps at training length, not OOD** — slow filter overfit
  training-length residual patterns. Would suggest trying different
  chunk_size values.
- **Destabilises training** — the +3.0 log_R init bias should
  prevent this, but watch the first few epochs' loss curves against
  Level 1.5's; they should overlap.

## Follow-ups if cascade shows a real effect

- **Chunk-size sweep** on lm200. `chunk_size ∈ {16, 32, 64, 128}`.
  Different resolution-vs-averaging tradeoffs; the sweep also
  informs the biological-timescale mapping (32 tokens ≈ what
  interval in "biological seconds"?).
- **K-level extension.** If two levels help, does three (fast →
  medium → slow, each at 32× the timescale) help more?
- **Alternative pooling.** Currently mean-over-chunk. Options:
  endpoint value only, or a learned linear pool.
- **Alternative broadcast.** Currently piecewise constant. Linear
  interpolation between consecutive D_slow_c values would smooth
  chunk-boundary jumps in θ̂.
- **Cascade on top of other filters.** InEKFCascade is written as a
  self-contained module; could similarly wrap `Level15EM`,
  `Level15NoDrop`, `Level15GSF`, etc.

## Key files in the current cascade implementation

- [model_inekf_cascade.py](model_inekf_cascade.py) — `InEKFCascade` +
  `MapFormerWM_Level15Cascade`
- [model_inekf_level15.py](model_inekf_level15.py) — imported for
  `assoc_scan_affine_scalar` (the scalar Hillis-Steele scan)
- [model.py](model.py) — inherited backbone (`MapFormerWM`,
  `WMTransformerLayer`, `ActionToLieAlgebra`)
- [train_variant.py](train_variant.py) — registered
  `"Level15Cascade"` in `VARIANT_MAP`
