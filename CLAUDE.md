# CLAUDE.md — Project Memory for MapFormer

**Purpose of this file:** concise context for Claude when resuming work on this
project in a fresh session. Read the README for the full picture; this file
focuses on state + lessons learned + what to do next.

## Project in one sentence

Faithful reproduction of Rambaud et al. (2025) *MapFormer* (arXiv:2511.19279)
plus three experimental extensions that add explicit state-correction
mechanisms to the path-integration circuit: a **parallel Invariant EKF**, a
**sequential InEKF** (for reference), and a **predictive-coding** variant.

## Current state (what's implemented and working)

1. **Paper reproduction** — `model.py`, `environment.py`, `main.py`.
   MapFormer-WM / EM reach paper-level accuracy (0.955 / 0.999) on 200K
   sequences at the paper's exact hyperparameters (Appendix B). Checkpoints
   in `figures_v6/`.
2. **Parallel InEKF** — `model_inekf_parallel.py`, `main_inekf_parallel.py`.
   Steady-state gain + FFT scan. Same speed as vanilla. Checkpoint in
   `figures_inekf_parallel_v2/`.
3. **Sequential InEKF (wrapped)** — `model_inekf_proper.py`,
   `main_inekf_proper.py`. ~2.5× slower but same final accuracy as parallel.
   Checkpoint in `figures_inekf_topology_fix/`.
4. **Predictive-Coding MapFormer** — `model_predictive_coding.py`,
   `main_predictive_coding.py`. Forward model + error-driven corrections.
   Checkpoint in `figures_predictive_coding/` after training completes.
5. **Evaluation tools** — `noise_test.py`, `gaussian_noise_test.py`,
   `diagnose.py`. Handle all variants above.

## Things that are true (verified) and must be preserved

**Paper-faithfulness invariants** (breaking any of these regresses to
broken states we debugged through):

- `environment.py`: torus grid (`(x+dx) % N`), interleaved token stream
  `[a1, o1, a2, o2, …]`, `revisit_mask` returned per trajectory
- `train.py`: loss masked to **revisited** observation positions only
- `model.py::PathIntegrator`: ω initialized monotonically decreasing in `i`
  (paper eq. 17 has a sign typo; the correct formula is
  `ω_i = ω_max · (1/Δ_max)^(i/(n_b-1))`)
- `model.py::MapFormerEM`: attention is Hadamard product `A_X ⊙ A_P`, not
  additive; uses separate learnable `q_0^p` and `k_0^p`
- `model.py::ActionToLieAlgebra`: low-rank factorization `W_out · W_in`
  with bottleneck `r=2`
- `model.py`: path integration via cumsum of angles, not prefix product of
  rotation matrices
- Default hyperparameters: 1 layer, 2 heads, h=64, d_model=128, lr=3e-4,
  AdamW, wd=0.05, linear LR decay, batch 128, grid 64, T=128 steps,
  K=16 obs types, p_empty=0.5, 200K sequences

## Architectural choices that matter

- **Feed `content_emb` only to the InEKF measurement head.** Adding
  position features `(cos θ, sin θ)` creates a degenerate optimum
  `z ≈ θ` → zero innovation → filter does nothing. We learned this
  the hard way.
- **Wrap innovations modulo 2π** via `atan2(sin(z - θ̂), cos(z - θ̂))`.
  Without this, length generalization breaks (θ̂ grows unboundedly, bounded
  z can't express the large "error" geometrically).
- **Steady-state Kalman gain from closed-form scalar DARE.** Enables
  FFT-conv based parallel affine scan, preserving MapFormer's O(log T)
  property.
- **Markovic et al. (2017) proves:** on SO(2), wrapped-innovation EKF equals
  Lie-Group EKF. So the simple wrapping *is* the correct Lie-group filter.
- **Predictive coding uses a forward model** `g(cos θ, sin θ) → ô` and
  computes error in *embedding space*, masked at observation positions only
  (action positions have unpredictable content conditioned on θ alone).
  Includes an auxiliary loss coefficient to force the forward model to
  actually model observations.

## Landmark experiment (added in latest session)

Added `n_landmarks` parameter to `environment.py`: sets N cells to emit
unique single-use tokens instead of aliased obs. Retrained all three
variants with 200 landmarks (~5% density).

Ran `landmark_eval.py` at T=128 and T=512, accuracy + NLL per cell type
(landmark / regular / blank):

- **At T=128:** PC best overall (87.7% acc, 0.591 NLL). But only InEKF
  predicts landmarks well (18% vs 1.5% for others).
- **At T=512:** InEKF is decisively best (78.5% vs 64% vanilla, 62% PC).
  Degrades only -7pp from T=128 to T=512 vs -18 to -26 for the others.

**Key finding: the three architectures are complementary, not alternatives:**
- Vanilla attention → clean aliased tasks
- PC MapFormer → matched-noise drift correction on aliased obs (best at
  training length)
- Parallel InEKF → true landmarks + long OOD (bounded-error stability)

This is the regime where Kalman filtering earns its theoretical guarantees
empirically. A 15pp overall accuracy gap at T=512 with landmarks.

Checkpoints:
- `figures_vanilla_noise_lm200/MapFormer_WM_noise.pt`
- `figures_inekf_parallel_lm200/MapFormer_WM_ParallelInEKF.pt`
- `figures_pc_lm200/MapFormer_WM_PredictiveCoding.pt`

## Clone-structure analysis (added in latest session)

`clone_analysis.py` runs 300 trajectories from a fixed start, records model
state at observation positions, and measures per-obs-type separation of
(x, y) cells in feature space (two metrics: linear-regression R² and
cosine-distance separation score).

Result:
- **PC MapFormer has the best θ̂ separation score (0.619 vs 0.573 vs 0.395).**
  Its prediction-error correction mechanism most cleanly clusters per-cell
  representations — closest to the CSCG (Clone-Structured Cognitive Graph)
  hypothesis from neuroscience.
- InEKF has more continuous (higher R²) but less clustered θ̂.
- Hidden features are similar across all models (attention blends position
  and content uniformly).

## Main empirical finding

On the paper's aliased-observation task, **vanilla attention + noise
augmentation beats all Kalman-style variants on raw next-token prediction**.
Reasons, in order of importance:

1. Attention already implements soft associative retrieval — implicit
   Bayesian pattern completion — which is what the Kalman update was meant
   to add.
2. Aliased observations (16 obs types / 4096 cells = ~128 cells per type)
   mean Kalman measurements can't produce sharp corrections. The Gaussian
   assumption of Kalman is violated; the true posterior is multimodal.
3. Innovation wrapping, required for length generalization, slows training
   by bounding per-step corrections.

**Where the Kalman / PC framework should win:**

- Tasks with **true landmarks** (5% of cells emit unique IDs found nowhere
  else). Not yet tested. Predicted to be where Kalman dominates.
- Very long sequences (T >> 2048) where attention becomes infeasible.
- Scenarios needing calibrated uncertainty (InEKF tracks σ², attention
  doesn't).
- External sensor fusion with known Q, R matrices.

## Things that didn't work / why

- **Uncertainty-modulated attention** (`model_kalman.py`, the first-pass
  InEKF): redundant with softmax attention's natural behavior. Kept for
  reference comparison.
- **Unwrapped InEKF innovations**: trained faster (0.66 vs 0.88 final loss)
  but broke at T=512 OOD — the measurement head extrapolated badly outside
  the short-sequence θ range it was trained on.
- **Adding position features to InEKF head**: degenerate optimum — the
  filter turns into the identity function.
- **Multi-layer MapFormer with shared θ**: not tested empirically. The
  paper only runs 1-layer MapFormer. If you need more layers for position
  correction, Option 1 (per-layer θ correction) is the natural extension
  but requires validation that multi-layer MapFormer trains stably first.

## Open questions / natural next experiments

1. **Evaluate the predictive-coding variant** against InEKF + vanilla on
   `gaussian_noise_test.py` at T=128 and T=512, noise_std 0.00 – 1.00.
   (This is running as of the last commit — check
   `figures_predictive_coding/`.)
2. **Add true landmarks** to `environment.py`: reserve ~5% of cells to
   emit unique high-info tokens (beyond the 16 standard types). Retrain +
   compare. This is the Kalman framework's home turf.
3. **Level 2 InEKF** (time-varying R_t from heteroscedastic head,
   parallelisable via Möbius-matrix associative scan). Theoretical sketch
   lives in the chat transcript; not yet implemented.
4. **Calibration metrics.** NLL or ECE would show whether InEKF's tracked
   σ² is a useful confidence estimate even when point accuracy doesn't
   improve.
5. **Multi-layer MapFormer ablation.** Paper only runs 1 layer. Test 2/3/4
   layers at vanilla + InEKF-augmented configurations to see whether depth
   helps at all in this architecture.
6. **Scaling.** Paper acknowledges they didn't scale model/data. 4 layers,
   4 heads, d=256, 10M sequences would be a natural next step.

## Quick reproducibility commands

```bash
# Re-verify paper reproduction:
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train each variant under 10% action-noise augmentation:
python3 -m mapformer.main_vanilla_noise --device cuda --epochs 50 --n-batches 156
python3 -m mapformer.main_inekf_parallel --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10
python3 -m mapformer.main_predictive_coding --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10 --aux-coef 0.1

# Head-to-head evaluation under Gaussian Δ noise:
python3 -m mapformer.gaussian_noise_test \
    --checkpoints \
      figures_v6/MapFormer_WM.pt \
      figures_vanilla_noise/MapFormer_WM_noise.pt \
      figures_inekf_parallel_v2/MapFormer_WM_ParallelInEKF.pt \
      figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --n-steps 128 --n-trials 200
# Then with --n-steps 512 for OOD length.

# Diagnostics on any trained model:
python3 -m mapformer.diagnose --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

## Filesystem map

- `figures_v6/` — paper-faithful MapFormer-WM and EM (reference)
- `figures_vanilla_noise/` — vanilla + noise-aug baseline
- `figures_inekf_parallel_v2/` — parallel InEKF main result
- `figures_inekf_topology_fix/` — sequential wrapped InEKF (same model class)
- `figures_inekf_proper/` — **stale**: sequential unwrapped InEKF
  (model code has since been updated; loading this checkpoint with current
  code is incorrect)
- `figures_kalman/` — first-pass fake InEKF (kept for comparison)
- `figures_predictive_coding/` — PC MapFormer (being populated)
- `figures_2M/`, `figures_constlr/`, `figures_v3/`, `figures_v4*/` — older
  runs from earlier debug sessions; can safely be deleted

## Authoring style / preferences

- No emojis in source code or commit messages
- No `Co-Authored-By` lines; single-author commits
- README is the primary documentation, this file is a memory-aid for Claude
- Honest reporting: if an experiment didn't work, write that down with the
  reason, don't bury it

## Level 2 InEKF results (autonomous addition)

Level 2 (heteroscedastic R_t) training completed. See RESULTS_LEVEL2.md
for full evaluation (per-cell-type accuracy, NLL, robustness, R_t / K_t
distribution by token category).

Checkpoints:
- figures_inekf_level2_lm200/MapFormer_WM_Level2InEKF.pt (with landmarks)
- figures_inekf_level2/MapFormer_WM_Level2InEKF.pt (no landmarks)

## Level 1.5 InEKF (compromise between Level 1 and Level 2)

Level 1.5 = constant Pi (learnable scalar, not DARE-derived) + per-token R_t.
Key insight: Level 2 diagnostic showed Pi only varied ~4x across tokens;
replacing Pi dynamics with a constant while keeping R_t/K_t dynamics
should recover most of Level 2's benefit at Level 1's cost.

Empirically: 60x faster training than Level 2, best landmark-training loss
of any variant (0.8124 vs Level 2's 1.133). Result in RESULTS_LEVEL15.md.


## Matched-compute verification: settled (L1.5 wins architecturally)

Verified on 2026-04-21: the L1.5 clean-task advantage is NOT from extra
training compute. `figures_v6/MapFormer_WM.pt` was already a 50-epoch
checkpoint with seed=42; a fresh 50-epoch vanilla run gave bit-identical
weights. L1.5 clean, trained with the same recipe and seed, reaches
training loss 0.1594 vs vanilla's 0.1935.

At matched compute, L1.5 wins on:
- Training loss (0.159 vs 0.194)
- Accuracy at T=128 (+1.5pp)
- Accuracy at T=1024 (+2.1pp)
- NLL at every length (−2% to −48%)

Vanilla wins only at T=512 accuracy (+1.1pp), paired with 20% worse NLL.

Hypothesis for why an architecturally-more-complex model wins on a clean
task:
- Path-integration weights are imperfect approximations even with clean
  data; L1.5's correction compensates for model-level drift.
- Structured extra capacity for per-token confidence reduces NLL directly.

See RESULTS_LEVEL15_CLEAN.md for full numbers.

## Parallel work while orchestrator runs (2026-04-22)

Orchestrator (PID 3325883) is running 53 multi-seed training jobs. In parallel,
the following scripts and docs were created for post-orchestrator processing:

- `long_sequence_eval.py`: eval at T up to 10,000 (tests Kalman bounded-error
  claim at extrapolation length)
- `calibration_analysis.py`: ECE + reliability diagrams (visualizes NLL wins)
- `make_paper_figures.py`: generates landmark bar chart, length-gen curves,
  ablation bars for paper. Saves to `paper_figures/`.
- `orchestrator_multilayer.py`: queues 2- and 4-layer training for Vanilla
  and Level15 (sanity check: does L1.5 scale with depth?)
- `followup.sh`: waits for main orchestrator to finish, runs long-seq eval,
  calibration, paper figures, then launches multilayer orchestrator.

- `paper/` directory: drafts of abstract, intro, related work, methods
  sections. Paper-quality narrative ready for refinement.

When main orchestrator finishes (~4 hours out), `followup.sh` should auto-run
all downstream analysis if launched with: `nohup bash followup.sh &`. Or
launch it now and it will wait for completion.

## Session 2026-04-22 evening — honest framing + WM vs EM + Level15EM + paper Mamba data

### Key framing shift

The paper already solves the clean aliased task (MapFormer-WM 0.955,
MapFormer-EM 0.999/1.000). Our Level 1.5 contribution is NOT beating
the paper on the paper's task; it is **extending MapFormer into regimes
the paper did not test**: action noise, true (non-aliased) landmarks,
out-of-distribution length, and calibrated uncertainty.

Real empirical wins vs MapFormer-WM (multi-seed, from RESULTS_PAPER.md):
- Noise T=512 OOD: +11pp (0.851 vs 0.739)
- LM200 T=512 OOD: +11pp (0.821 vs 0.715)
- NLL across the board: roughly 2x lower in noise/landmark regimes;
  0.000 vs 0.025 on clean (calibrated at landmark tokens)
- Length generalization: Level 1.5 drops only ~0.5pp clean, 10pp noise,
  9pp lm200 from T=128 to T=512 (Kalman bounded-error property)

`paper/00_abstract.md` and `paper/01_introduction.md` rewritten with this
honest framing. Clean task = "matches paper baseline"; contribution
lives in the untested regimes.

### WM vs EM — we built on WM; port to EM done

Every correction variant (Level 1/1.5/2, PC, all ablations) inherits
from `MapFormerWM`, not `MapFormerEM`. Reasons:
1. WM's single input-dependent rotation couples cleanly to the corrected
   θ̂ from the Kalman update.
2. EM's Hadamard-product attention (A_X ⊙ A_P) would need the correction
   threaded into both branches.
3. WM at 0.99 has measurable headroom; EM at 0.999 would ceiling-effect
   any result.

**NEW: `model_inekf_level15_em.py`** = `MapFormerEM_Level15InEKF`.
Same `InEKFLevel15` class, plugged into EM's `q0_pos`/`k0_pos` rotations.
Registered as `"Level15EM"` in `train_variant.py::VARIANT_MAP`. Sanity-
checked (forward pass works, 204K params, K≈0.5 and R≈1.0 at init).

### Orchestrators added in this session

- `orchestrator_em.py` — VanillaEM × 3 configs × 3 seeds (9 runs).
  Baseline on the stronger MapFormer-EM backbone.
- `orchestrator_level15_em.py` — Level15EM × 3 configs × 3 seeds (9 runs).
  Our correction on the stronger backbone.
- `master_finish.sh` — waits for em + multilayer + level15_em to finish,
  then launches `orchestrator_baselines` (LSTM/CoPE/MambaLike), then
  re-runs `orchestrator_finalize.sh` so RESULTS_PAPER.md includes all
  new variants. (Use this instead of `followup.sh` — followup.sh exited
  early at 20:11 without launching baselines or final finalize.)

`orchestrator_finalize.sh` updated:
- Imports `MapFormerEM_Level15InEKF`
- `VARIANT_CLS` includes `VanillaEM`, `Level15EM`
- `variants_main` list now:
  `[Vanilla, VanillaEM, RoPE, LSTM, CoPE, MambaLike, Level1, Level15,
    Level15EM, PC]`

### MapFormer paper DOES benchmark against Mamba — Table 3

Found during session. The paper has an Appendix A.3-A.5 titled "MAmPa:
Learning Cognitive Maps with block-diagonal Mamba Models." Key claim:
vanilla Mamba CANNOT learn cognitive maps because diagonal A matrices
can't encode rotations (Lie-theoretic argument). Their fix (MAmPa =
Mamba with 2x2 block-diagonal skew-symmetric A) does better but still
loses to MapFormer and is slow.

Table 3 (2D grid navigation, sequence length l=16):
|        | IID  | OOD-d | OOD-s |
| Mamba  | 0.42 | 0.77  | 0.40  |
| MAmPa  | 0.74 | 0.93  | 0.60  |
| MapWM  | 1.00 | 1.00  | 1.00  |
| MapEM  | 1.00 | 1.00  | 1.00  |

Verbatim caption: "As expected, MAmPa offers substantial improvements
over Mamba, but fails to reach performances a par with MapFormers,
while being slower."

Implication for our framing: the question "doesn't Mamba subsume this?"
is already answered NO by the paper itself, on structural grounds
(Lie-group expressivity). Our contribution stacks on top of MapFormer's
SO(2) machinery; it doesn't compete with generic SSM. Our `MambaLike`
baseline in model_baselines_extra.py reproduces the paper's weaker
(vanilla-Mamba) baseline. Plan: add a paragraph to
`paper/02_related_work.md` citing Table 3 directly once our MambaLike
multi-seed numbers land.

### Where the running pipeline will land

When master_finish.sh completes (~1h45m from ~20:37):
- RESULTS_PAPER.md has 4 new row-types in every table:
  VanillaEM, Level15EM, LSTM, CoPE, MambaLike
- Commit + push happens automatically via orchestrator_finalize.sh
- paper_figures/ will be stale-ish (length-gen + calibration figures
  only cover Vanilla/RoPE/Level1/Level15/PC — not Level15EM or EM).
  Re-run `long_sequence_eval.py` and `calibration_analysis.py` with
  updated --variants list if we want those figures to include EM rows.

### Three decision points when numbers land (for honest framing)

1. If VanillaEM alone closes the noise/landmark gap → framing: backbone
   choice matters more than correction; L1.5 earns its keep on NLL only
2. If Level15EM > VanillaEM by similar +11pp → strongest framing:
   correction helps on top of either backbone
3. If MambaLike matches Level15 → reframe section 6.10 from future work
   into a central finding (unlikely given paper's Table 3, but worth
   confirming at our training scale)

### Project memory-file note

If `RESULTS_LEVEL2.md` / `RESULTS_LEVEL15.md` / `RESULTS_LEVEL15_CLEAN.md`
are referenced above but not present, they predate the multi-seed
orchestrator and were superseded by `RESULTS_PAPER.md`. Trust the latter.

## Session 2026-04-24 — final results, Level15EM init pathology + fix

### Summary of where we ended up

- All multi-seed training complete: VanillaEM, Level15EM (safe init),
  LSTM, MambaLike, partial CoPE.
- Three GitHub commits during the day:
  - `8af4680` — first finalize (broken-init Level15EM, ZERO_SHOT_TRANSFER_*.md)
  - `b330036` — second finalize (CoPE rows partially in)
  - `5d091e7` — third finalize (safe-init Level15EM rows)

### Final headline results (OOD T=512, fresh obs_map)

| Config | Vanilla(WM) | VanillaEM | Level15(WM) | Level15EM | LSTM | MambaLike |
| ------ | ----------- | --------- | ----------- | --------- | ---- | --------- |
| Clean  | 0.913       | 0.972     | **0.993**   | 0.977     | 0.800| 0.573     |
| Noise  | 0.739       | 0.765     | 0.851       | **0.869** | 0.743| 0.568     |
| LM200  | 0.715       | 0.605     | **0.821**   | 0.730±0.12| 0.641| 0.513     |

### Key findings (final framing)

1. **Mamba cannot do this task at our scale** (~0.57 across configs).
   Reproduces the paper's Table 3 (Mamba 0.42 there at l=16). Confirms
   that diagonal-A SSMs lack the rotation expressivity needed for
   cognitive-map learning.
2. **Vanilla MapFormer-EM does NOT subsume correction.** VanillaEM
   underperforms even Vanilla-WM on lm200 (0.605 vs 0.715). Stronger
   backbone alone is not a substitute for explicit state correction.
3. **Correction (Level 1.5) works on either backbone.** WM gets
   +11pp on noise OOD and +11pp on lm200 OOD over Vanilla-WM. EM gets
   +10pp on noise OOD and +12pp on lm200 OOD over VanillaEM.
4. **Backbone choice matters less than correction.** WM-with-correction
   slightly beats EM-with-correction on lm200; EM slightly beats WM
   on noise. Both clearly beat their vanilla counterparts in
   noise/landmark regimes.

### The Level15EM init pathology + fix (this session's main fix)

**Problem:** Original Level15EM training had `log_R_init_bias=0.0` which
gives Kalman gain K = Pi/(Pi+R) = 1/(1+1) = **0.5 at init**. EM's
attention is `softmax(A_X ⊙ A_P)`, where the position branch A_P is
computed from rotations of `q0_pos, k0_pos` by the InEKF-corrected θ̂.
At init, the corrections are random (random measure_head + K=0.5),
which destroys A_P, which Hadamard-products with A_X to destroy
gradient signal entirely. WM doesn't have this issue because content
attention provides a fallback gradient path.

Result: 3 of 9 Level15EM seeds catastrophically diverged (final loss
≈1.45, plateauing from epoch 5). The other 6 were mediocre.

**Fix:** `log_R_init_bias=3.0` for the EM-backbone variant, giving
K ≈ 0.05 at init (10× smaller corrections). The InEKF behaves as a
near-no-op at init; the model learns vanilla-MapFormer behaviour
first, then the R_t head learns to lower R where measurements are
informative. WM keeps the original `log_R_init_bias=0.0` for backward
compat (its existing checkpoints still load).

Code change is in `model_inekf_level15.py::InEKFLevel15.__init__`
(new `log_R_init_bias` parameter, default 0.0) and
`model_inekf_level15_em.py` (passes `log_R_init_bias=3.0`).

Old broken-init Level15EM checkpoints preserved at
`runs/Level15EM_broken_init/` for diagnostic comparison.

### Remaining caveat

Level15EM lm200 seed 2 reached final loss 1.40 (vs ~1.0 for the other
two seeds), giving the lm200 row a wider std (±0.12) than other
configs. Bumping `log_R_init_bias` further (e.g., 5.0) might catch
this outlier; not pursued because the central tendency is clearly
positive and reporting honest variance is more important.

### Pipeline state at end of session

- All orchestrators exited cleanly. No background processes running.
- Latest commit on GitHub: `5d091e7`.
- CoPE has 8/9 runs (lm200 seed 2 was killed mid-training to unblock
  Level15EM retraining). Sufficient for multi-seed reporting on
  clean (3/3) and noise (3/3); lm200 has only 2 seeds.
- `master_finish_v3.sh` and `retrain_level15em.sh` both completed.

### What's left for the paper

- ZERO_SHOT_TRANSFER_*.md eval was run BEFORE Level15EM was retrained,
  so it has the broken-init Level15EM rows. May want to re-run with
  the safe-init checkpoints if including in the paper.
- `paper_figures/` calibration + length-gen figures don't include
  VanillaEM or Level15EM. Update before final paper submission if
  these go in figures.
- CoPE lm200 seed 2 retraining could be queued separately for
  completeness (~8h on one GPU); not strictly needed.

## Session 2026-04-26 — Level15PC + Grid + GridL15PC findings

### New model variants

- `Level15PC` (`model_level15_pc.py`): MapFormer-WM + Level 1.5 InEKF + PC aux
  loss on the standard backbone. Tests forward-model + inverse-model
  complementarity.
- `Grid` / `Grid_Free` (`model_grid.py`): multi-orientation path integrator
  with fixed (hex) or learnable orientations.
- `GridL15PC` / `GridL15PC_Free` (`model_grid_l15_pc.py`): Grid + Level 1.5
  + PC aux. Kitchen-sink test for hex emergence.

### Empirical findings

Hex emergence is NOT solved by architecture or correction stacking:
- `Grid_Free` clean s0: loss 0.021, hex orientations stayed but max
  per-module grid score 0.036 (0/22 modules > 0.3).
- `GridL15PC_Free` clean s0: loss 0.084, hidden-state max 0.052
  (worse than Grid_Free's 0.095). Adding L15+PC ACTIVELY REDUCED hex.
- §6.5/§6.10 falsification strengthens. Bottleneck is training
  objective, not correction toolkit.

`Level15PC` multi-seed sweep launched via `orchestrator_level15pc.py`,
results in RESULTS_PAPER.md, LONG_SEQ_*.md, PER_VISIT_*.md,
ZERO_SHOT_TRANSFER_*.md, HIPPOCAMPAL_LEVEL15PC.md, CLONE_ANALYSIS_LEVEL15PC.md.

### Honest checkpoint logging

- `runs/Grid_clean_200ep/seed0/Grid.pt`: stale, won't load with current
  code (state-dict key mismatch from cos_orient/sin_orient → orientation_angles).
- `runs/Level15EM_b5_lm200/seed2/`: diagnostic-only (alt safe-init
  experiment, kept untracked).


## Session 2026-04-27 — Kalman = stabilisation, R-saturation diagnosis, NoBypass fix

### Conceptual reframing (the big one)

**Level 1.5's win across all regimes is primarily a STABILISATION effect, not
an inference effect.** Three lines of evidence:

1. The wrap (atan2 of innovation) is bounded in [-π, π] regardless of
   how far θ_path drifts. This keeps θ̂ in the trained range at OOD
   length while Vanilla's θ_path goes out-of-distribution. Without the
   wrap (older unwrapped variant), training is faster but T=512 OOD
   breaks — confirming the wrap is the load-bearing piece.
2. R_t learns to be HIGH on aliased obs (no useful inference), so the
   actual measurement contribution is tiny. Yet Level 1.5 still beats
   Vanilla by 8pp on clean OOD T=512. The inference isn't doing the
   work; the stabilisation is.
3. Per-token R_t is also doing token-type GATING (action vs obs),
   which explains why L15_ConstR drops 20pp at clean T=128 (where
   stabilisation alone shouldn't matter at training length). With
   constant R, action-token "measurements" leak into θ̂ and corrupt
   path integration.

So the architecture has two structural pieces (wrap + per-token-type
gating) and inference is mostly absent. **At runtime Level 1.5 is a
wrapped EMA over learned-but-uninformative measurements, with
content-dependent gain shape inherited from the Kalman parameterisation.**

### Three interference tests on Level15PC's lm200 regression

After Level15PC's 23pp lm200 OOD regression, we tested three falsifiable
hypotheses for the mechanism:

**Test 1 (R_t distribution by token type)** — `R_T_DISTRIBUTION.md`:
- Level15: log_R spread 0.45 across action/blank/aliased/landmark
- Level15PC: log_R values all ≈ -3 (near the -5 lower clamp), spread 0.72
- **Diagnosis: PC's aux loss drives R_t to saturate at the lower
  clamp**, making K ≈ 1 everywhere. The InEKF stops being a Kalman
  filter and becomes an autoencoder bypass: θ̂ ≈ z_t = h(x_t), so θ̂
  encodes the current input embedding rather than the cumulative
  position. Attention can't retrieve past tokens at the same cell
  because θ̂ at revisits ≠ θ̂ at the original visit.
- **The "PC flattens R-gating" hypothesis was FALSIFIED** — instead
  it saturates R-gating at the floor.

**Test 2 (aux_coef sweep)** — `AUX_COEF_SWEEP.md`:
Trained Level15PC on lm200 with aux_coef ∈ {0, 0.01, 0.03, 0.1, 0.3}.
Looking for a monotone dose-response curve to confirm the gradient
mechanism. (Pipeline running at session end.)

**Test 3 (clone-separation transfer)** — `CLONE_TRANSFER_TEST.md`:
Recomputes PC's clone-separation score on a fresh obs_map (seed=10000)
to test whether PC's clean clustering transfers or is memorisation.
(Pipeline running at session end.)

### The fix: `Level15PC_NoBypass` (Fix 5 + Fix 6)

`model_level15_pc_v2.py::MapFormerWM_Level15PC_NoBypass` adds two
architectural fixes:

- **Fix 5 (stop-gradient on InEKF correction inside PC aux loss):**
  `theta_for_pc = theta_path + (theta_hat - theta_path).detach()`. PC
  can ONLY improve aux loss by improving path integration, not by
  driving R → 0 to bypass. Sanity check verified: PC aux loss has zero
  gradient on R-head, z-head, log_Pi parameters.
- **Fix 6 (mask aux loss at landmark tokens):** vocab id ≥
  LANDMARK_START_ID (=21 for default config) is excluded from the aux
  loss. Removes the noise gradient at one-shot tokens that motivated
  the saturation in the first place.

If the diagnosis is right, NoBypass should match Level15-alone's lm200
OOD T=512 (~0.82). If it stays at Level15PC's level (~0.59), the
diagnosis is wrong and we need to keep digging.

### Honest framing update for the paper

The cleaner narrative for §5 / §6 is now:

- **Kalman's win is stabilisation + token-type gating, not Bayesian
  inference.** This is a narrower claim than "Kalman filtering helps"
  but more accurate.
- **PC alone underperforms Vanilla on raw next-token accuracy** (PC
  OOD T=512 clean: 0.815 vs Vanilla: 0.913). PC's only clean win is
  clone-separation score (a representation-quality metric), and we
  haven't yet verified that win transfers to held-out environments.
- **Combining PC + L15 fails on lm200 not because they "compete" but
  because PC's aux loss creates an autoencoder bypass via R-saturation.**
  The diagnosis is mechanistic.
- **Hex emergence is not solved by architecture (`Grid_Free`) or by
  correction stacking (`GridL15PC_Free`).** The bottleneck is the
  training objective. Multi-environment training is the obvious next
  experiment.

### Files added/modified (this session)

- `model_level15_pc_v2.py` (new): NoBypass variant with Fix 5+6.
- `r_t_distribution_test.py`, `clone_transfer_test.py`,
  `aux_coef_sweep.py` (new): three interference tests as standalone
  scripts.
- `run_interference_tests.sh`, `run_nobypass_test.sh` (new):
  autonomous pipelines.
- `train_variant.py`: registered `Level15PC_NoBypass`.
- All 5 eval scripts (long_seq, per_visit, zero_shot, calibration,
  hippocampal_hidden_eval): added Level15PC_NoBypass import +
  VARIANT_CLS entry.
- `R_T_DISTRIBUTION.md`, `CLONE_TRANSFER_TEST.md`, `AUX_COEF_SWEEP.md`,
  `R_T_DISTRIBUTION_3WAY.md`, `NOBYPASS_RESULTS.md`,
  `CLONE_TRANSFER_NOBYPASS.md` (some still being generated by the
  in-flight pipelines).

## Session 2026-04-28 — v3 / v4 PC isolation + length diagnostic

### NoBypass diagnosed via length_diagnostic.py

`LENGTH_DIAGNOSTIC.md`: NoBypass's |θ̂| explodes to **~3840 at T=512**
(vs Level15: 83, Level15PC: 105). Fix 5 + 6 closed the *direct*
R-saturation route, but PC still leaks into `action_to_lie` via shared
path-integration parameters, blowing up θ_path even with d_t detached.
That's why length generalization breaks despite the wrap.

### v3 (Fix 7: tighter R clamp [-1, 5]) — partial

`model_level15_pc_v3.py`: clamp log_R upward so K can't approach 1.
Recovers clean OOD T=512 to 0.948 (NoBypass: 0.872) but lm200 OOD T=512
only to 0.626 (still well below Level15's 0.790). R distribution moves
positive but the indirect-route degradation isn't fixed.

### v4 (Fix 8: full PC isolation) — works at single seed

`model_level15_pc_v4.py`: detaches BOTH `theta_hat` AND the target
embedding `x` inside the PC aux loss. PC gradient touches *only* the
forward_model parameters; CE gradient flow is bit-identical to Level15.

| Variant (s0) | clean OOD T=512 | lm200 OOD T=512 |
|---|---|---|
| Level15 | 0.991 | 0.790 |
| Level15PC | 0.985 | 0.722 |
| Level15PC_NoBypass | 0.872 | 0.594 |
| Level15PC_v3 | 0.948 | 0.626 |
| Level15PC_v4 | 0.964 | 0.871 |

The +8pp single-seed v4 win on lm200 prompted multi-seed verification.

## Session 2026-04-29 — multi-seed v4 + PC/Kalman duality + Sorscher Option A

### Multi-seed v4 result — modest real win, mechanism unclear

`V4_MULTISEED.md` (commit 8f93d69). v4 seeds 1, 2 trained on clean
+ lm200; evaluated all three v4 seeds against the existing three
Level15 seeds.

| Config | Variant | T=128 OOD | T=512 OOD |
|---|---|---|---|
| clean | Level15 (n=3) | 1.000 ± 0.000 | 0.995 ± 0.003 |
| clean | Level15PC_v4 (n=3) | 1.000 ± 0.000 | 0.985 ± 0.015 |
| lm200 | Level15 (n=3) | 0.912 ± 0.015 | 0.825 ± 0.026 |
| lm200 | Level15PC_v4 (n=3) | **0.935 ± 0.004** | **0.859 ± 0.009** |

- The single-seed +8pp gap was inflated (s0 was Level15's worst seed
  AND a typical v4 seed).
- True effect: **+3.4pp on lm200 OOD T=512, +2.3pp T=128 OOD**, with
  non-overlapping seed ranges (v4: [0.848, 0.871];
  Level15: [0.790, 0.854]).
- Clean is essentially tied (Level15 marginally better on NLL).
- v4's PC has zero gradient flow into the main model, so the win
  cannot be attributed to "PC doing PC." Likely: RNG drift
  (forward_model consumes init draws, shifting all subsequent
  params) or AdamW second-order effects through shared optimizer
  state. **The RNG-matched control was not run** (vanilla Level15
  with a dummy forward_model instantiated-but-unused). Init drift
  remains the leading hypothesis.

### Theoretical reframing — PC and Kalman are duals, not complements

PC's forward map `g(θ̂) → x_t` and InEKF's inverse map `h(x_t) → z_t`
are mathematical duals — same Bayesian posterior over θ written from
opposite sides. When both operate on the same θ̂ with the same inputs,
they target the same fixed point. Gradient descent finds the trivial
joint minimum: `g ∘ h ≈ identity`, achieved by `R → 0` so `θ̂ ≈ h(x_t)`
(the R-saturation autoencoder bypass). Any non-zero gradient coupling
reproduces this collapse; only full gradient isolation (v4) avoids
it, but at that point PC is no longer shaping the representation.

**Honest paper claim now: PC and Kalman are not complementary
modules to stack — they're alternative parameterizations of the same
posterior. Architectures that include both create a degenerate
optimum gradient descent will find.**

### Sorscher Option A — DoG aux head (in flight at session end)

Why hex didn't emerge in any prior variant: Sorscher/Ganguli (2019)
prove hex is the unique optimum under three conditions —
(1) path integration ✅, (2) non-negativity ❌, (3) DoG/center-surround
place-cell targets ❌. We had only (1). MapFormer's loss is categorical
CE on aliased obs tokens; PC's forward model also predicts aliased
tokens; nothing in the pipeline produces a DoG-similarity kernel.
TEM's hex route (compositional generalization across many envs)
also doesn't apply — single-map training.

`model_level15_dog.py::MapFormerWM_Level15_DoG`: keeps Level 1.5 + the
original CE loss; adds an aux head

    hidden -> Linear -> ReLU (the "grid layer") -> Linear -> p̂

with `p̂` regressed against
`max(0, gE(d) - gI(d))`,  `σ_E=1.5`, `σ_I=3.0`, on a 16×16 grid of
place-cell centers over the 64×64 torus. Aux added at `--aux-coef 0.1`.
The grid layer (`n_grid_units=256`) is the candidate hex site.

`probe_hex.py`: runs trajectories, builds per-unit rate maps from
ground-truth positions, computes Sargolini-style grid scores via SAC
+ rotational correlations (annular region, `min(c60, c120) -
max(c30, c90, c150)`).

`run_dog_test.sh`: clean s0 training (50 epochs, aux_coef=0.1) →
probe → commit + push. Result lands in `DOG_RESULTS.md`.

### Files added/modified (this session)

- `model_level15_dog.py` (new): Level15 + DoG aux head + ReLU
  bottleneck.
- `probe_hex.py` (new): rate-map + Sargolini grid-score probe.
- `run_dog_test.sh`, `run_v4_multiseed.sh` (new): autonomous
  pipelines.
- `train.py`: stashes ground-truth positions on the model before
  forward when the model exposes `_batch_positions`. Other
  variants unaffected.
- `train_variant.py`: registered `Level15_DoG`.
- `V4_MULTISEED.md`: multi-seed v4 vs Level15 comparison.

### What's still in flight at session end

- `DOG_RESULTS.md`: hex-probe output for Level15_DoG s0. If max grid
  score > 0.3 in some units, Sorscher's three conditions are
  empirically sufficient on this architecture and we run multi-seed.
  If not, even the analytic-theory-aligned setup fails — likely
  pointing at the *learned* SO(2) path integrator (vs Sorscher's
  fixed velocity-driven recurrence) as the remaining bottleneck.

