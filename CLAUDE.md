# CLAUDE.md — Project Memory for MapFormer

**Purpose of this file:** concise context for Claude when resuming work on this
project in a fresh session. Read the README for the full picture; this file
focuses on state + lessons learned + what to do next.

## Project in one sentence

Faithful reproduction of Rambaud et al. (2025) *MapFormer* (arXiv:2511.19279)
plus three experimental extensions that add explicit state-correction
mechanisms to the path-integration circuit: a **parallel Invariant EKF**, a
**sequential InEKF** (for reference), and a **predictive-coding** variant.


## IMPORTANT (2026-08-27): the MapFormer paper is now at v4 and DID language

### v4 CHANGED THE REPRODUCTION TARGET -- our cited numbers are v1

CLAUDE.md's "paper Table 2 verbatim" (MapWM 0.99/0.99/0.96, MapEM-os 1.0/0.99/0.97)
is the **v1** table. **v4 Table 2, 2D columns:**

| model | IID | OOD-d | OOD-s |
|---|---|---|---|
| MapWM-r2 | 1.00+/-0.00 | 1.00+/-0.00 | 0.99+/-0.01 |
| MapEM-os | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| MapEM-s  | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |

Our measured 0.989 (WM) / 0.987 (EM) MATCHED v1 but is marginally BELOW v4.
Do not claim "matches the paper" without saying which version.
Other v4 changes: error bars throughout; MapWM split into **r1/r2** (our default IS
r2); **TAPE and PathAtt added as baselines**; RoPE(4L) 2D IID 0.33 -> 0.82;
Mamba 0.42/0.77/0.40 -> 0.38/0.66/0.30 and MAmPa 0.74/0.93/0.60 -> 0.84/0.96/0.71
(our Table 3 quotes are v1). **The omega sign typo SURVIVES into v4** (now eq. 18,
printed with a negative exponent and n_b rather than n_b-1, contradicting its own
stated boundary condition) -- our fix remains correct and necessary.

### A paper-reported negative we did not have
**MapWM COLLAPSES in 5D**: 0.75/0.50/0.35, *worse than CoPE* (0.94/0.80/0.69).
MapEM-s holds (1.00/1.00/0.87). v4 Table 6.

### Selective RoPE (ICLR 2026) is the same primitive, discovered independently
arXiv:2511.17388 (Movahedi et al., ELLIS/EPFL/Freiburg) posted **21 Nov 2025**;
MapFormer posted **24 Nov 2025**. **Neither cites the other in ANY version**
(verified across all bibliographies). Both cite PaTH. Selective RoPE is **published
at ICLR 2026**; MapFormer v4 is still marked "Preprint".
Mechanism: `omega = temp*cumsum(conv1d(W_omega@q)); rope(q,k,...)` vs MapFormer's
`theta = cumsum(omega * W_out W_in x_t)`. Real differences: angle from **query** vs
**token**; **no rank bottleneck** vs rank-r (load-bearing: r=1 -> 0.66 in 2D);
**conv1d** vs none; **sigmoid gate + bias + weight norm** vs none; commutativity is
MapFormer's central axis and the word appears **0 times** in Selective RoPE;
SSM/state-transition framing vs deliberately-not-SSM; linear attention + softmax vs
softmax only. No MapEM analogue exists on their side.

### NoPE: the navigation side of this literature has none
Selective RoPE runs NoPE everywhere and **NoPE sometimes WINS** (GLA 1.3B avg acc:
NoPE 55.2 > SRoPE 54.6 > RoPE 54.4). **MapFormer has zero NoPE baselines.** Ours is
built and registered (`NoPE`, model_baseline_nope.py, param-identical to RoPE).


This file was written against **v1 (Nov 2025)**. The paper is at **v4 (10 May 2026)**;
**Sec 5.5 "Scalability to Natural Language" + App B.5 are NEW**:
- 12-layer MapWM on OpenWebText, ~10^11 tokens, 4xH100, 5 seeds:
  **RoPE 19.14+/-0.14 vs MapWM 18.79+/-0.15** (consistent ppl win, p<0.005)
- **BLiMP: NO gain** (0.78 vs 0.79) -- "gains do not come from better syntactic
  modeling, which might require another mechanism than (commutative) path-integration"
- **Length extrapolation LOSES to CoPE and PathAtt** on NarrativeQA
- The paper itself frames MapFormer-on-language as the SAME FAMILY as CoPE/PaTH
- It nominates **code modeling** as the natural next test

Consequence: do NOT run MapWM-vs-RoPE on language. Also note **Selective RoPE
(ICLR 2026, arXiv:2511.17388) is essentially our exact mechanism on language**
(theta = temp*cumsum(omega) -> rope), and **PoPE's rotation angle is NOT
content-dependent** (only its magnitude is) -- PoPE is not in this family.
Full landscape, numbers and open slots: **LANGUAGE_LANDSCAPE.md**.


## STANDING RULES 8-12, all bought by the 2026-08-26..28 retractions

Run `python3 -m mapformer.experiment_audit --runs-dir <dir> --control <inert-twin>
--control-of <real-arm>` BEFORE interpreting any run directory. It checks 8-11
automatically in ~30s. Four claims were retracted that week; every one is caught by
that script.

**Rule 8 -- MEASURE the noise floor; never assume it.** Train an arm that is PROVABLY
function-identical to a real one (same params, effect multiplied out, zero gradient)
and report the gap. Measured here: **mean |delta| 0.150, range -0.23..+0.41** on
MiniWorld. Most effects chased that week were smaller than that. No effect below the
measured floor is reportable.

**Rule 9 -- check whether accuracy is just the training loss.** Measured r = **-0.996**
over 57 runs (acc = 1.039 - 0.555*loss). When |r| > 0.98 the held-out eval carries no
information the loss does not, every "effect" is a loss gap in disguise, and the only
honest analysis is the loss-matched residual.

**Rule 10 -- verify CONVERGENCE (loss slope over the final 10%), and check the LR
SCHEDULE.** `LinearLR(1.0->0.0)` decays from step one with no warmup: on a
plateau-then-cliff landscape a run can never escape the plateau late, so the budget
measures "did the transition fire early", not "can this model solve the task".
Switching to 5% warmup + cosine-to-10% moved one arm from **0.448 to 0.990 on the
same task** and INVERTED the sign of the headline effect. Never compare unconverged arms.

**Rule 11 -- "null" requires power.** Compute the minimum detectable effect
(2.8*sd/sqrt(n)). At n=9 with sd 0.18 the MDE is **0.165** -- larger than any component
effect in the study. Say **"unmeasured"**, not "null", unless MDE < the effect you are
dismissing. Also: conditioning on convergence can SELECT INTO A CEILING (every
surviving pair already >0.95), which shows ~0 by construction -- report the threshold
sensitivity, never a single cutoff.

**Rule 12 -- put the seeds on the comparison you are CLAIMING.** Seeds were run on
"MapPoPE vs RoPE" and the claim made was "MapPoPE composes, i.e. beats its
COMPONENTS" -- a different comparison, at n=1, with a margin under half the noise.
Before writing a sentence, ask which two columns it compares and whether THOSE have
seeds.

**Corollary to rule 5, learned the hard way:** check your own GATE data against your
proposed mechanism. The "attention horizon" story was falsified by gate G6 -- revisit
lags SHORTEN with grid size (47/43/38/33) and the fraction inside the horizon RISES
(0.43->0.50) -- data collected before training and never cross-checked.


## Current state (what's implemented and working)

1. **Paper reproduction** — `model.py`, `environment.py`, `main.py`.
   Reproduces the paper's 2D grid-navigation accuracy. CORRECTED 2026-08-09:
   this entry previously cited "paper-level accuracy (0.955 / 0.999)"; those
   are NOT the paper's numbers. Paper Table 2 (1D-2D grid navigation), 2D
   columns, verbatim: MapWM IID 0.99 / OOD-d 0.99 / OOD-s 0.96; MapEM-os IID
   1.0 / OOD-d 0.99 / OOD-s 0.97. Our measured held-out revisit accuracy at
   T=128 (PAPER_TASK_ACCURACY.md, n=3 seeds): WM 0.989 +/- 0.010, EM (single
   p_0) 0.987 +/- 0.012 (best seed 0.9995). Our EM = the paper's MapEM-os
   (both observation and structure; paper sec. C: "MapEM-os relying on both
   observation and structure to compute attention"). Checkpoints in
   `runs/paper_task/`. NOTE: separate q0_pos/k0_pos is PAPER-FAITHFUL (App. A.4:
   "our MapFormers use two separate initial vectors k0p and q0p ... we suspect
   this separation to be beneficial"). Single-p_0 is an ABLATION of that stated
   suspicion, and it refutes it (0.987 +/- 0.012 vs 0.898 +/- 0.108). Do NOT
   delete the separate-q0/k0 EM results; both configurations are reportable.
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


## Session 2026-04-30 — Fix 8 audit, RNG-control, MiniGrid wrapper, SE(n) gap analysis

### Fix 8 audit confirmed v4 has airtight gradient isolation

Ran a per-parameter gradient trace on Level15PC_v4:
- Aux loss gradient hits ONLY `forward_model.*` params (norm sum 0.062)
- ALL other params (token_emb, action_to_lie, omega, transformer layers,
  out_norm, out_proj, log_Pi, log_R_head, measure_head) get gradient
  ONLY from CE (norm sum 8.86)

So v4's surprise +3.4pp lm200 OOD T=512 win over Level15 is NOT from
gradient leakage. Three remaining candidates: (a) RNG state shift from
forward_model's ~50K extra init params, (b) clip_grad_norm coupling
through joint norm including forward_model's small but nonzero grad,
(c) statistical noise across 3 seeds.

### Control experiment queued (run_v4_control.sh)

Trains Level15PC_v4 architecture with `--aux-coef 0.0`. forward_model
exists (controls for RNG shift) but never gets gradient (forward_model
contributes 0 to grad-clip joint norm). 3 seeds × 2 configs queued —
waiting for the user's `tma_standalone` jobs to free GPUs, then runs
autonomously and pushes V4_CONTROL_RESULTS.md.

Decision rule: if Control ≈ v4 → win is RNG-shift only (a).
If Control ≈ Level15 → win is from aux loss / clip-coupling (b).

### What the MapFormer paper actually does for higher dimensions

Searched the original paper for SE(2) / continuous-task / SE(3) mentions:
NONE. They handle 3D/5D grids by stacking SO(2) blocks ("translations
in n dimensions are just n independent 1D-translations"). They briefly
explore non-commutative groups (4D rotations, MapEM-NC) but stay within
rotation groups. They never use translation×rotation Lie groups.

§7 Limitations admits causal-only, didn't scale, WM-vs-EM on reasoning.
Does NOT mention continuous-state navigation, action noise, or SE(n).

So our SE(n) extension is genuinely unexplored relative to the paper.
The paper-narrative is honest: "MapFormer establishes input-dependent
SO(2) for 2D grids; we push into noise/landmarks/calibration regimes
the paper didn't test, and propose SE(n) generalisation as the natural
next step."

### MiniGrid wrapper added (`minigrid_env.py`)

Goal: extend our toy torus benchmark to a real published navigation
benchmark for the deployment story.

`MiniGridWorld` adapter exposes the same `generate_trajectory()`
interface as `GridWorld`, returning `(tokens, obs_mask, revisit_mask)`
in our token format. Plug-and-play with `train.py` / `train_variant.py`.

Current design:
- 7 discrete actions (left, right, forward, pickup, drop, toggle, done)
- Obs tokenization: just the cell directly in front of the agent
  (image[3, 5] in MiniGrid's 7x7 egocentric view)
- Three tokenization modes: `obj_only` (11 types), `obj_color` (66),
  `full` (~200)
- Random policy by default (matches random-walk paradigm)
- Action noise via `p_action_noise` (matches torus convention)
- Revisit defined by `(x, y, direction)` tuple

Smoke-tested on `MiniGrid-Empty-8x8-v0` (vocab 19) and
`MiniGrid-DoorKey-8x8-v0` with obj_color tokenization (vocab 74).

Natural progression: Empty-8x8 → DoorKey-8x8 (has key+door = natural
landmarks) → KeyCorridor → ObstructedMaze. Lift to harder envs as
results validate the wrapper.

Next experiments (not yet run):
1. Train Vanilla MapFormer + Level 1.5 on MiniGrid-Empty-8x8 with
   action noise. Compare revisit accuracy at OOD length.
2. DoorKey is the natural lm200 analogue — door+key are unique
   landmarks. Test landmark exploitation in a real-task setting.

### Files added/modified this session

- `minigrid_env.py` (new): MiniGridWorld adapter
- `run_v4_control.sh` (new): aux_coef=0 control pipeline
- This CLAUDE.md update


## Session 2026-05-01 — DoG bug, continuous nav, stochastic-transition framing

Heavy discussion + infra session, mostly waiting on shared GPUs. Full thread
in `SESSION_2026-05-01.md`. Key landings:

### Bug discovery (important)
DoG kernel in `model_level15_dog.py` (and the new `continuous_nav.py`) used
unnormalised Gaussians: `max(0, exp(-d²/2σE²) - exp(-d²/2σI²))`. At d=0 both
Gaussians equal 1 → target = 0 → ReLU keeps it 0 → **target is silently all
zeros**. The earlier `DOG_RESULTS.md` (max grid score 0.036) was on broken
targets — vacuous, not a real Sorscher test. Fixed to use normalised
Gaussians (1/σ² prefactor); targets now correctly show ~0.33 at centre.
`DOG_RESULTS_FIXED.md` re-run pending GPU.

### MiniGrid pipeline (cached + RoPE diagnostic)
- `MiniGridWorld_Cached`: pre-built 25K-trajectory buffer, ~35× speedup
  (1.7s/epoch vs 360s live). Trade-off: same trajectories reused across
  epochs; cached/live numbers differ by ~2pp on clean.
- DoorKey-8x8: Vanilla 0.916 / Level15 0.900 clean OOD T=512 (basically
  tied — small env, drift sub-cell). Level15 +10pp on noise.
- DoorKey-16x16 + MultiRoom-N4-S5: registered in train_variant.py.
- Long-T eval to T=2048 (`MINIGRID_DOORKEY_LONGT.md`): NLL gap opens
  meaningfully even when accuracy ties; noise-accuracy gap grows with T
  (+16pp at T=2048).
- RoPE diagnostic (`MINIGRID_DOORKEY_ROPE_DIAG.md`): RoPE collapses at
  long T (0.834 → 0.699 from T=512 → T=2048 clean). Validates the env
  exercises path integration.

### Continuous 2D nav infrastructure (Cueva/Wei/Sorscher)
- `continuous_nav.py`: SE(2) state on torus, velocity commands with
  Gaussian process noise, DoG-of-position obs targets.
- `model_continuous.py`: Vanilla + Level15 with optional ReLU bottleneck
  (`n_grid_units > 0`) for hex probing.
- `train_continuous.py` + `probe_hex_continuous.py` + `eval_continuous.py`:
  full pipeline. In flight (waiting on GPU).

### Stochastic-transition MDP framing (the action-noise reframe)
Action-token corruption is **mathematically equivalent to a
stochastic-transition MDP** for uniform policies. Both produce identical
(action_record, observation) data distributions. Use the
stochastic-transition framing in any writeup — standard control/RL
vocabulary, much harder to dismiss as artificial.

`environment.py` now exposes `--p-transition-noise` (genuine
execution-time stochasticity, distinct from `--p-action-noise`'s
post-hoc record corruption). Empirical equivalence will land in
`STOCHASTIC_TRANSITION_RESULTS.md`.

### NLL > accuracy as the more discriminating metric
Two models can be tied on accuracy with NLL differing 2×. Level 1.5
dominates NLL across all regimes — calibration matters even when point
predictions are similar. Most cognitive-map papers don't report NLL;
push it as a primary metric.

### Defensible regime-by-regime claim
- Empirically validated: long-horizon clean OOD (+8pp on torus T=512),
  heteroscedastic landmarks (+11pp), stochastic-transition / proprioceptive
  noise (+10pp), calibration (NLL 2× lower across the board), long-T
  noise OOD on MiniGrid (+16pp at T=2048).
- Theoretically motivated, untested: animal nav, cheap-IMU robotics,
  multi-sensor fusion with known noise.
- Honest negatives: small-env clean (sub-cell drift, no advantage),
  categorical obs noise (attention's job), aliased multi-modal posteriors
  (attention's job).

### Files added/modified this session
- `continuous_nav.py`, `model_continuous.py`, `train_continuous.py`,
  `probe_hex_continuous.py`, `eval_continuous.py` (new)
- `minigrid_env.py`: `MiniGridWorld_Cached` class (was already added,
  refined this session)
- `model_level15_dog.py`: DoG kernel bug fix (normalised Gaussians)
- `environment.py`: `p_transition_noise` parameter for stochastic-transition
- `train.py` / `train_variant.py`: thread `p_transition_noise` through
- `run_dog_fix_and_continuous.sh`: unified auto-pipeline (P1 DoG fix +
  P2 continuous nav + P3 stochastic transition)
- `SESSION_2026-05-01.md`: full discussion thread for cross-chat sync

### Pending at session end
`run_dog_fix_and_continuous.sh` is polling for free GPUs (other user has
both pegged at 100% on py-tbfm). When it fires it'll produce
`DOG_RESULTS_FIXED.md`, `CNAV_*.md`, `STOCHASTIC_TRANSITION_RESULTS.md`,
auto-commit + push.

## Session 2026-05-10 — TEM fixes, β / dropout discovery, EM/WM mechanism, goal-directed

### Bug fixes (load-bearing)

- **TEMFaithful predict-then-update bug.** Old order queried memory with
  PRE-action g (the wrong cell's content). Fixed by updating g via W_a
  BEFORE prediction. lm200 OOD T=512: 0.42 (chance) → **0.969**. Reverses
  the prior session's "TEMFaithful is the worst baseline" claim — it's now
  the lm200 leader.
- **TEM-t NaN.** Unconstrained `ReLU(e · W_a)` recurrence → `||e||` grows
  ~10× per 8 steps → 1e13 by L=255 → NaN. Fix: add `e_pre_attn` LayerNorm
  (paper-faithful pre-attention) AND `e_in_rnn` LayerNorm inside the
  recurrent loop (deviates from paper; replaces sensory-landmark reset
  which our random-walk setup lacks).
- **TEMFaithful unconstrained W_a NaN.** `exp(skew(A_a))` orthogonal
  parameterisation. Same matrix-exp-of-skew used elsewhere.

### Headline discovery: post-attn residual dropout, not β, was load-bearing

Tested "learnable softmax temperature β" (`Level15Beta`) as the lightest
way to close the gap to TEMFaithful on lm200. It worked: 0.819 → 0.935
(+12pp). BUT learned β values barely moved from init (0.148–0.182 vs init
0.125 = 1/√d_head). A 1.2–1.5× sharpening cannot explain +12pp.

`WMTransformerLayer` (baseline) vs `WMTransformerLayer_Beta` had TWO
differences:
1. β: learnable temperature on Q·K^T (init at 1/√d_head).
2. Post-attention residual: original wraps `o_proj(out)` in `self.dropout`;
   Beta drops the wrapper.

`Level15NoDrop` ablation (fixed β, only dropout removed): **0.948 ± 0.025**
on lm200 OOD T=512 — matches Beta. Dropout removal was load-bearing; β
was a red herring.

Regime-dependent Pareto trade-off:
- Clean: −0.7pp acc, NLL **doubles** (calibration loss).
- Noise: +2pp acc, NLL −6%.
- LM200: **+12pp acc, NLL −56%.**

Mechanism: Vaswani's default block dropout regularises when retrievals
are redundant (aliased obs has ~128 copies; feature-zeroing averages out)
and destroys when they're rare (a landmark token appears once). For
paper: frame as Pareto-shift, NOT strict improvement.

### EM-vs-WM mechanistic story

- **EM:** `A = softmax(A_X ⊙ A_P)` — multiplicative AND-gate.
- **WM:** combined additively in the score — OR-gate.

| Regime | A_X | A_P | Winner | Observed |
|---|---|---|---|---|
| Aliased + short (paper main) | Noisy | Sharp | EM | EM > WM ✓ |
| Aliased + large vocab + short (paper Fig 4c) | Noisier | Sharp | EM | EM ≫ WM (per paper) |
| Aliased + long OOD (ours) | Noisy | Drift-degraded | WM | WM > EM ✓ |
| Landmarks (rare unique content) | Sharp | Drift-degraded | WM | WM 0.715 > EM 0.605 ✓ |
| Landmarks + correction | Sharp | Repaired | Both helped | L15-WM 0.821 > L15-EM 0.730 ✓ |

Backbone ordering is regime-dependent. "EM is the better model" is a
paper-task claim, not universal.

**RETRACTED 2026-08-09 -- the table above does not survive its own data.**
Two of its five rows are unsupported:
- "Aliased + long OOD -> WM > EM" is contradicted by the repo's OWN clean row
  (VanillaEM 0.972 > Vanilla 0.913 at OOD T=512) and now by the multi-seed vocab
  sweep, where at n_obs=16 / T=512 EM beats WM on 3/3 seeds (VOCAB_SWEEP_MULTISEED.md).
- "Landmarks -> WM 0.715 > EM 0.605" rests on lm200 checkpoints voided by the
  RETRACTION section below.
What the vocab sweep actually shows: EM's deficit is VOCABULARY-specific
(n_obs=256, reproducible at +/-0.020), not length-specific. The AND-gate story
may still be right, but the regime axis in this table is wrong, and the paper's
Fig 4c direction (EM better at LARGE vocab) is not reproduced either -- EM's
relative position gets worse from n_obs=16 (+0.027) to n_obs=256 (-0.086).
n_obs=4096 is degenerate (all models at the 0.500 blank floor) and carries no
signal in either direction.

### Paper scaling claims (verified via WebFetch)

Figure 4: EM > WM along (a) head size at l=256, (b) sequence length up
to l=384, (c) vocab up to 10000 at l=16. **None test long-l + rare-content,
where our results flip the ordering.**

### New files / variants

- `model_inekf_level15_beta.py` (`Level15Beta`): learnable β.
- `model_inekf_level15_nodrop.py` (`Level15NoDrop`): only post-attn
  residual dropout removed. The clean ablation.
- `model_inekf_gsf.py` (`Level15GSF`): Gaussian Sum Filter with K parallel
  Level 1.5 chains. K learnable `θ_init_k` offsets, cumulative-log-
  likelihood mixture weights. Smoke-tested + registered. **Not yet trained.**
- `environment_goal.py`: `GoalDirectedGridWorld` + `bfs_torus`. Episode =
  `[goal_token, T_explore random walks, T_navigate BFS-optimal]`.
- `train_goal.py`: CE on next-action prediction at navigate-phase positions.
  Chance = 0.25; smoke test (3 epochs Vanilla) → 0.708 held-out accuracy.

### Vocab sweep result (`VOCAB_SWEEP_RESULTS.md`, single seed, T=512 OOD)

| Variant | n_obs=16 | n_obs=256 | n_obs=4096 |
|---|---|---|---|
| Vanilla | 0.862 | 0.665 | 0.470 |
| VanillaEM | 0.968 | **0.562** | 0.495 |
| Level15 | 0.991 | 0.980 | 0.456 |
| Level15EM | 0.986 | 0.970 | 0.411 |

At our l=128/T=512 OOD, paper's "EM wins at large vocab" claim does
NOT invert the ordering — VanillaEM actively *crashes* at n_obs=256
(0.562, worse than Vanilla 0.665). Correction rescues both backbones to
near-parity (Level15 0.980 ≥ Level15EM 0.970). At n_obs=4096 all
collapse to ~0.45 — degenerate regime (each cell ≈ unique token,
test-env obs_map is totally different; uninformative).

**Conclusion: at long l, vocab scaling does NOT flip the WM-EM ordering.
Paper's Fig 4c is l=16-specific.**

### Goal-directed result (`GOAL_DIRECTED_RESULTS.md`, single seed, lm200)

| Variant | T_exp=32, T_nav=32 | T_exp=64 (train) | T_exp=128 OOD |
|---|---|---|---|
| Vanilla | 0.628 | 0.950 | **0.766** |
| Level15 | 0.939 | 0.947 | **0.950** |
| Level15EM | 0.936 | 0.949 | 0.948 |
| Level15NoDrop | 0.939 | 0.946 | 0.949 |

Headline: Vanilla cognitive maps degrade with longer explore (drift
accumulates → action selection breaks at T_exp=128: -18pp). Correction-
stabilised maps STAY navigable across all explore lengths — the bounded-
error Kalman promise made concrete on a behavioural task.

- Level15 vs Vanilla: +18pp at OOD explore length. Correction is
  decisive for goal-directed use of the cognitive map.
- Level15EM ≈ Level15: tied. The multiplicative AND-gate is benign when
  correction repairs A_P enough that the gate fires reliably.
- Level15NoDrop ≈ Level15: dropout removal has NO effect on
  goal-directed task (action prediction is 4-class — retrieval is
  dense, not rare-signal-dependent).

This is the cleanest cognitive-map utility test we've done — connects
path-integration correction to navigation behaviour.

### Pending decisions

- GSF launch: depends on vocab sweep + goal-directed results. The dropout
  finding weakens the "multimodal Bayes is the missing piece" story.
- Level15NoDrop multi-seed on clean + noise (currently only lm200 has
  3 seeds; needed to nail down the Pareto trade-off cleanly).

### NoDrop multi-seed (`NODROP_PARETO_RESULTS.md`)

| Config | Vanilla | Level15 | Level15NoDrop |
|---|---|---|---|
| Clean T=512 | 0.911 | 0.993 | 0.985 (−0.8pp, within std) |
| Clean T=512 NLL | 0.458 | 0.039 | 0.070 |
| Noise T=512 | 0.638 | 0.702 | 0.699 (tied) |
| LM200 T=512 | 0.716 | 0.819 | **0.948** |
| LM200 T=512 NLL | 1.391 | 0.897 | 0.317 |

NoDrop is essentially Pareto-equivalent on clean/noise (differences
within seed std) and a +13pp win on lm200. Stronger than the
Pareto-shift framing suggested; this is closer to "near-free win for
landmark tasks."

### DoorKey-8x8 BC (`DOORKEY_BC_RESULTS.md`)

| Variant | match-acc | closed-loop success |
|---|---|---|
| Vanilla | 0.875 | 0.250 |
| Level15 | 0.875 | 0.230 |
| Level15EM | **0.938** | 0.190 |
| Level15NoDrop | 0.812 | 0.240 |

**EM wins on match-acc here, OPPOSITE of our torus result.** Mechanism-
consistent: DoorKey is egocentric (only the cell directly in front
visible), so A_X is much noisier than torus → EM's multiplicative
AND-gate filters that A_X noise → backbone ordering flips. Same
prediction, different sign because the regime flipped.

Closed-loop success ~0.20 across all variants is the BC distribution-
shift ceiling. DAgger queued to break it.

### GSF (`GSF_RESULTS.md`, n=3 on lm200) — ⚠ RETRACTED

> **The Level15 row (0.819) is a non-converged April checkpoint.** Retrained
> under current code Level15 reaches **0.996**, beating TEMFaithful (0.982).
> The "GSF closes 95% of the TEM gap" reading is an artifact: there was no gap.
> All lm200 tables below share this defect — see the RETRACTION section at the
> end of this file and `CORRECTED_LM200_LEADERBOARD.md`.

| Variant | lm200 T=512 OOD | NLL |
|---|---|---|
| Level15 | 0.819 ± 0.025 | 0.897 |
| Level15NoDrop | 0.948 ± 0.025 | 0.317 |
| Level15GSF | **0.956 ± 0.042** | 0.227 |
| TEMFaithful | 0.969 ± 0.010 | 0.171 |

K=8 parallel Kalman chains with cumulative-log-likelihood mixture
weighting closes 95% of the TEMFaithful gap. Multi-modal Bayesian
filtering actually works. Combined with NoDrop result: **two
independent fixes each ~match TEMFaithful's lm200 lead**.

### Linear probe of frozen prediction-trained models (`PROBE_GOAL_RESULTS.md`)

| Variant | Train-probe acc | Held-out probe acc |
|---|---|---|
| Vanilla | 0.592 | **0.555** |
| Level15 | 0.634 | 0.630 |
| Level15EM | 0.649 | 0.631 |
| Level15NoDrop | 0.640 | 0.637 |

Frozen backbone + single linear head → action. **+7.5pp gap from
Vanilla → Level15 in the FROZEN representation** (held-out probe).
Cognitive maps differ in CONTENT, not just trainability. This is the
cleanest "Level15 builds a richer cognitive map" claim — no goal-
directed training of the backbone, just a linear readout.

### Five-finding paper synthesis (current state)

1. **Prediction baseline**: Level15 beats Vanilla on acc + NLL across
   all regimes (multi-seed, existing).
2. **Pareto-shift (NoDrop)**: one inherited dropout removal → +13pp on
   lm200, ~free on clean/noise (multi-seed, new).
3. **Multi-modal Bayes (GSF)**: K=8 chains closes 95% of TEMFaithful
   gap on lm200 (multi-seed, new).
4. **Goal-directed BC**: correction stabilises cognitive map under OOD
   explore-length on torus (+18pp); EM wins on partial-obs envs (DoorKey)
   and ties on full-obs (torus) — mechanism predicts both signs.
5. **Frozen-probe**: cognitive maps differ in CONTENT, not just
   trainability. Linear readout from Level15 representations carries
   +7.5pp more goal-directed info than from Vanilla.

### Queued at end of session (will auto-commit)

- `run_gsf_nodrop.sh` (PID 228104): GSF + NoDrop combo → `GSF_NODROP_RESULTS.md`.
- `run_gsf_modes.sh` (PID 228699): GSF mode-weight diagnostic → `GSF_MODES_DIAGNOSTIC.md`.
- `run_dagger.sh` (PID 228706): 4 variants × 4 DAgger rounds on
  DoorKey-8x8 → `DAGGER_RESULTS.md`. Tests whether richer cognitive
  maps yield better RECOVERY from off-expert states.

### Honest caveats (do NOT bury in the paper)

- DoorKey closed-loop success 0.19-0.25 across all variants — BC
  distribution-shift ceiling. Architecture differences show in
  match-acc, not closed-loop behaviour. DAgger should reveal whether
  the cognitive map difference cashes out in actual recovery.
- Vocab=4096 collapse for ALL variants — degenerate regime, no signal.

### New files this session

`model_inekf_level15_beta.py`, `model_inekf_level15_nodrop.py`,
`model_inekf_gsf.py`, `model_inekf_gsf_nodrop.py`,
`environment_goal.py`, `train_goal.py`, `doorkey_solver.py`,
`train_doorkey_bc.py`, `train_doorkey_dagger.py`, `probe_goal_linear.py`,
`probe_gsf_modes.py`, plus the corresponding `run_*.sh` pipelines.


## RETRACTION: all lm200 results (2026-07-16)

**Every lm200 table in this file, RESULTS_PAPER.md and README.md is invalid**
and is being regenerated. Read this before citing any landmark result.

### What happened

Stored lm200 checkpoints trained **2026-04-22..24** (Vanilla, Level15,
Level15EM, VanillaEM, RoPE, PC, Level1, CoPE, LSTM, MambaLike) **never
converged**: final CE loss ~1.0 instead of ~0.005. Checkpoints trained
**2026-05-08+** (TEMFaithful, Level15GSF, Level15_SR, NoDrop, Vanilla_ExtraHead)
converged normally. The reported lm200 "leaderboard" is monotonic with
training convergence, not with architecture:

| Reported rank | reported acc | stored final loss |
|---|---|---|
| Vanilla | 0.716 | 1.22 (stuck) |
| Level15 | 0.819 | 1.01 (stuck) |
| NoDrop | 0.948 | 0.24 (partial) |
| GSF | 0.956 | 0.0007 (converged) |
| TEMFaithful | 0.969 | 0.0004 (converged) |

### Scope — lm200 ONLY

Clean and noise checkpoints retrain **bit-identically** to the stored ones
(`NOISE_CLEAN_REVALIDATION.md`), so those results are VALID. Root cause is the
landmark-cell-selection RNG (`rng.permutation(n_cells)[:n_landmarks]`), which
only runs when `n_landmarks > 0`; lm200 training is basin-sensitive to the
resulting layout. Level15-WM's own code is byte-identical April vs now.

### Corrected ranking (fresh, current code, seed 0)

Level15 **0.996** > TEMFaithful 0.982 > NoDrop 0.915 > Level15EM 0.860 >
Vanilla 0.835 > VanillaEM 0.807 > PC 0.721 > MambaLike 0.567 > RoPE 0.513

### Claims REVERSED

- "TEMFaithful is the lm200 leader" — false; Level15 beats it.
- "NoDrop +13pp over Level15 on lm200" — reversed.
- "GSF / multiple fixes close the TEM gap" — no gap existed.
- "Level15Cascade wins lm200" — artifact (see hierarchy work).

### Claims that SURVIVE (strengthened)

- Level15 >> Vanilla on lm200: 0.996 vs 0.835 (~+16pp, larger than reported).
- RoPE / Mamba collapse is genuine: fresh RoPE 0.513 (stored 0.523), fresh
  MambaLike 0.567 — they reproduce, so cognitive-map necessity holds.

### Rule going forward

Never compare a freshly-trained variant against a stored baseline checkpoint;
retrain every arm in the same batch. Run `validate_task.py` before new tasks.


## Session 2026-07-25 — koopman clone, compositional multi-seed, model rename

Ran on a second server (koopman, 2×RTX 4090). Cloned the repo, hit two env
gotchas now documented in HOURGLASS_README: pip grabbed torch+cu130 on a
CUDA-12.4 driver (silent CPU fallback) → installed torch 2.6.0+cu124; and the
package `__init__.py` needs full `requirements.txt` (matplotlib/scipy/sklearn),
not a torch+numpy subset.

### Compositional Hourglass — multi-seed (n=3) OVERTURNS the single-seed read

`run_comp_multiseed.sh` + `agg_comp_multiseed.py` (new): 6 variants × seeds
{0,1,2}, train T=256, eval fresh env (seed=10000) at T∈{256..2048}. Added two
non-MapFormer controls (`PlainHourglass`/`PlainFlat` = ordinary index-RoPE, no
path integration). Table in `COMPOSITIONAL_MULTISEED.md`; findings in
`COMPOSITIONAL_EXPERIMENT.md` (RESULTS). `COMPOSITIONAL_RESULTS.md` (single-seed)
is marked SUPERSEDED.

Headline (cross_nb_acc, the compositional target):
- **Hierarchy helps in BOTH backbones** (Hourglass > flat on all 3 seeds, paired,
  for MapFormer AND plain). H2 (fixed-stride absolute-θ Hourglass ≈ flat)
  FALSIFIED — it beats flat.
- **MapFormer barely helps over a plain transformer.** Flat: MapWM 0.270 vs
  Plain 0.213 (~+0.05, consistent). Hierarchy: MapWM-Hier vs Plain-Hier is
  seed-dependent (paired Δ −0.01/+0.29/+0.02 — all from seed1). `MapEM-Flat`
  (0.097) is WORSE than plain (0.213): the EM AND-gate hurts compositional
  transfer. The relayed "plain ≈ 0.06 chance floor" prediction is FALSIFIED —
  the action stream is in the input, so a plain transformer path-integrates via
  attention; MapFormer's SO(2) code is an inductive bias, not privileged info.
- **MapWM-Hier is high-variance** (seed1 outlier 0.625 vs ~0.30 for seeds 0,2;
  std > gap). The clean, low-variance version of "hierarchy helps" is the plain
  family. More MapWM-Hier seeds is the key open follow-up.
### Hierarchy on text, MapFormer family (2026-08-28) -- NULL on quality, real on cost

`ENWIK8_HIERARCHY.md`. **MapWM-Hier 1.4537 vs MapWM-FlatHG 1.4506 bpc** at EXACT
parameter parity (28,371,016 both -- same 3-block scaffold, differing only in whether
the middle block pools k=2), 36k iters, dim 880, deterministic val. Hierarchy is
**+0.0032 WORSE**, inside the 0.003-0.007 checkpoint sd -> a null on bpc. The only
measurable effect is compute: **1.23x throughput (20.10 vs 16.36 it/s measured alone
on an idle GPU), -14.1% peak memory**, matching a -17.4% analytic FLOP count. An
earlier "-8.6% wall time" figure from the training run was contaminated by GPU
co-tenancy -- do not quote it. Note the saving is a LINEAR win (half the tokens
through one block's FFN), not the quadratic-attention win the hourglass is sold on:
attention is only 8.8% of a block at d=880/L=512, so the saving grows only to -21.7%
at L=8192, against a -25% ceiling for this 1-of-3-blocks scaffold.

Same DIRECTION as the plain-family run (1.4844 vs 1.4727, +0.0117 worse at -18.75%
FLOPs), now with param parity and deterministic val. Two families agree: on text,
hierarchy is an EFFICIENCY property, not a quality win.

n=1 per arm, so only an effect much larger than 0.007 was detectable -- consistent
with a null, not proof of one (rule 11). Says nothing about compositional transfer
or long-horizon aggregation, where hierarchy's actual wins live; next-byte
prediction is exact-recall, the regime where a lossy summary is not a sufficient
statistic and hierarchy is EXPECTED to lose.

The two PoPE arms in that run (MapPoPE-Hier 1.4591, PoPE-Hier 1.4553) are
EXPLORATORY: MapPoPE-Hier silently trained at r=2 while the MapWM arms trained at
r=4 (the `_widen_to_d` rank bug, fixed in e8f8f50 -- not retroactive), and neither
has a flat control. The primary pair is unaffected: both MapWM, both r=4.

- enwik8 scaffold (Gate B): **CORRECTED 2026-08-28** -- the '≈2.00 vs ≈2.07,
  hourglass better' figure is WRONG and no saved data supports it. Actual, at
  identical params (31,787,264) and seq 2048: **hourglass 1.4844 vs flat10
  1.4727 -- hourglass is WORSE by +0.0117**, with -18.75% FLOPs and -17.6%
  wall-time. The earlier partial run agrees (1.5099 vs 1.4973, worse by
  +0.0126). This is an EFFICIENCY result (equal-ish quality at less compute),
  NOT a quality win, and must not be listed among hierarchy's wins. Equal
  params, seq=2048 — efficiency property reproduces.
- **Phase 2 (`Hourglass_MotifSeg`, H3) still NOT built** — the room-boundary-
  segmented motif-collapsing variant is the predicted *real* hierarchy win.

### Model rename (backbone × structure), non-breaking

Added aliases in `train_variant.py::VARIANT_MAP` (old keys still resolve, so
existing checkpoints and the other server's names keep working):
`MapWM-Flat`=Vanilla, `MapEM-Flat`=VanillaEM, `MapWM-Hier`=Hourglass_k2,
`MapWM-FlatHG`=HourglassFlat3, `Plain-Hier`=PlainHourglass, `Plain-Flat`=PlainFlat.
`agg_comp_multiseed.py` renders these display names.

Position-encoding note (the experimental variable): `MapWM-Hier` rotates q,k by
the **path-integration** angle `θ=ω·cumsum(Δ(actions))` (RoPE *mechanism*, not
index RoPE); `Plain-Hier` uses standard index RoPE `θ=t·freqs`. The two
Hourglass models are identical scaffolds differing only in the rotation angle.

### Infra lesson

A background run launched via the tool's run_in_background died when the parent
session process exited (orphaned children killed). For multi-hour runs use
`setsid`/`nohup` so they survive session teardown; they then won't send a
harness completion notification (check the `.done` marker / log instead).

### Phase 2 (H3) built + run — oracle motif-segmentation does NOT help

`MapFormerWM_Hourglass_MotifSeg` (`Hourglass_MotifSeg` / alias `MapWM-MotifSeg`,
commit a905c51): identical to `Hourglass_k2` (same 600,917 params) except it pools
on ORACLE room boundaries (env `meta.new_room` -> per-token seg id, threaded via
`model.wants_seg_id`/`_batch_seg_id`) instead of a fixed token stride. Causal
(verified, max leak 4.8e-7), param-parity exact.

Result (n=3, `run_motifseg.sh`): cross_nb_acc **0.254 ± 0.014** at T=256 — BELOW
the flat control (MapWM-FlatHG 0.281) and far below MapWM-Hier; 2nd-worst
hierarchy variant. Trains fine (exact_acc 0.943). **Not a bug** — failure is
specific to compositional transfer.

Why: v1 tests segmentation ALIGNMENT only; it omits the LOCAL-FRAME-RESET (H3
ingredient 3). The coarse room-summary means MapFormer hidden states that still
carry ABSOLUTE path angle, so identical motifs at different locations do NOT
collapse to one code — the sufficient statistic is never formed, and ~8-token/
256-step compression is pure loss. So this falsifies "room-aligned pooling helps"
but NOT "collapse-by-structure helps". Decisive remaining test = v2 with the
frame-reset. Prior on v2 lowered (even the oracle upper bound landed below flat).

Phase 2 v2 (frame-reset, `model_hourglass.py`: `MapWM-MotifSeg-FR`,
`MapWM-Flat-FR`, commit 410e316): added the local-frame-reset (verified it
zeroes the angle at room entry so identical motifs collapse). **H3 DECISIVELY
FALSIFIED** — it made BOTH metrics worse: cross_nb_acc 0.157 (below v1's 0.254
and below plain), exact_acc 0.94->0.77. The predicted trade-off didn't happen;
both dropped. Destroying absolute position breaks the cognitive map cross-
instance retrieval ALSO needs (over-aliases — the failure mode MapFormer's
position code exists to prevent). MotifSeg-FR ~= Flat-FR (reset dominates,
hierarchy irrelevant). Collapse-by-construction is HARMFUL, not just unhelpful.
The real synergy lives on the hier-goal task instead (next section).

### The two-question summary (for the paper / "what does each addition buy")

WM and hierarchy help ORTHOGONAL metrics:
- **WM (path integration) -> exact positional recall, growing with length.** vs
  plain RoPE, exact_acc gap widens with T (hier: +0.03/+0.10/+0.18/+0.17 over
  T=256..2048). Barely helps compositional; MapEM-Flat is WORSE than plain.
- **Hierarchy -> compositional transfer, backbone-independent.** vs flat, cross_nb
  ~+0.11 (clean in plain, high-variance in MapFormer); ~0 on exact_acc. Generic
  multi-scale compression, NOT task-structure alignment (MotifSeg didn't help).

### Hierarchical goal-directed navigation — RETRACTED 2026-08-09

> **This entire section is void, and with it the "MapFormer x hierarchy synergy"
> claim.** The hier-goal task never measured navigation: randomising the goal AND
> the whole explore phase leaves accuracy unchanged (0.912 -> 0.913), and
> closed-loop success is 0.013-0.037 against a 0.010 random floor with the BFS
> oracle at 1.000. An n-gram on the ACTION STREAM ALONE scores 0.969 (order 1,
> raw BFS) and 0.971 (order 3, the interleaved "fix"). See HIERGOAL_ABLATION.md.
>
> **What the hierarchy evidence actually says**, all of it predating this result:
> hierarchy buys COMPOSITIONAL TRANSFER and long-horizon AGGREGATION, and costs a
> little on PRECISE RETRIEVAL.
>   wins  — compositional motif 0.415 vs flat 0.270 (and Plain 0.318 vs 0.216);
>           aggregate task T=2048 0.537 vs 0.401; enwik8 -18.75% FLOPs at
>           SLIGHTLY WORSE bpc (1.4844 vs 1.4727) -- efficiency, not quality
>   loses — HIER_ATTN_LONGT T=4096 0.769 vs flat 0.861; ROUTE_ATTN 0.764 vs 0.849;
>           SPACETIME_HIER 0.833 vs 0.955; Match-Query 0.786 vs 0.888
> Oracle room-aligned pooling did NOT help (0.254 vs flat 0.281), so it is not
> about segmentation alignment. The super-additive-interaction claim is gone; the
> two-question summary above it was right all along.

#### (original text, retained for the record)

New task (`environment_hier_goal.py`, `train_hier_goal.py`, `validate_hier_goal.py`):
`[room_goal, local_goal, explore, navigate(BFS)]`, fixed anchor -> absolute
position, hierarchical goal (room+local) needs both scales, eval at OOD explore
length. Motivated by the analysis that MapFormer (multi-scale position) and the
time-hierarchy help ORTHOGONAL things unless a task demands absolute position at
multiple scales AT ONCE over a long horizon.

Result (n=3, `HIERGOAL_RESULTS.md` / `HIERGOAL_MULTISEED.md`): at OOD explore
length **MapWM-Hier is best by a wide, reliable margin** (T=128 acc 0.907±0.026
vs MapWM-Flat 0.656, Plain-Hier 0.700, Plain-Flat 0.548). The 2x2 **interaction
is consistently +0.09-0.10** across OOD lengths -> genuine super-additivity (the
compositional task had ZERO interaction). MapWM-Hier is also uniquely stable OOD
(±0.03-0.07 vs ±0.08-0.21). In-distribution (T=64) all four tie ~0.96 -- the
effect is entirely OOD-length.

Honest correction: the single-seed scan showed Plain-Hier collapsing (0.48) and
I called it "hierarchy hurts plain" -- that was seed noise; at n=3 hierarchy
helps plain slightly too. Real story = super-additive interaction, not a sign
flip. Caveats: n=3, one task, OOD-only. This is the "true combination" answer:
it took the TASK creating the multi-scale-position demand, not a cleverer arch.

## Session 2026-08-09 — Match-Query verified; three task lines voided

### The one result that survived everything

**Path integration is necessary for in-context cognitive maps** (`MATCH_QUERY_SCALE.md`).
Match-Query: explore with observations revealed, then continue BLIND (observations
withheld) and predict the observation at each cell.

| variant | 64^2 (n=5) | 128^2 (n=3) | TQ=2048 |
|---|---|---|---|
| MapWM-Flat (path int.) | 0.730 +/- 0.247 | 0.823 +/- 0.043 | 0.693 |
| PlainFlat (index) | 0.154 +/- 0.018 | 0.192 +/- 0.022 | 0.093 |

Chance 0.0625. No seed overlap (worst PI 0.398 vs best index 0.178). The axis is
path integration, NOT the encoding: MapPoPE-Hier (PoPE + path int.) 0.847 vs
PoPE-Flat (PoPE + index) 0.117. Boundary: at n_obs=4 the per-seed separation
BREAKS (one PI seed 0.321 inside the index range 0.321-0.345).

Trustworthy because it passed the **context-destruction ablation**: 0.918 -> 0.074
(explore obs shuffled) -> 0.076 (query path shuffled). hier-goal on the same
manipulation went 0.912 -> 0.913.

### What was voided today

- **hier-goal, both versions.** Solvable from the action prefix. My interleave
  "fix" moved the shortcut from order 1 (0.969) to order 3 (0.971) and I validated
  only order 1. Closed-loop 0.013-0.037 vs a 0.010 random floor.
- **ALL FOUR remaining planner tasks** (`PLANNER_TASK_AUDIT.md`): goal 0.969,
  rooms_goal 0.969, rooms_maze 0.791, maze_varying 0.650 -- n-grams on the action
  stream alone, chance 0.250. 13 result files upgraded SUSPECT -> VOID, including
  the +7.5pp frozen-probe result that was a headline finding.
- **The WM-vs-EM regime narrative**, and its proposed replacement mechanism
  (A_P kernel geometry), falsified on a pre-registered test.
- **The lap mechanism claim.** Lap training collapses Match-Query (-0.293), but
  removing the reward -- the whole lap-counting demand, one token per episode --
  gives -0.291. It is catastrophic forgetting under distribution shift. The
  theta-drift metric is also NOT diagnostic: a model WITH a working map scores
  0.252/4.37, the lap model 0.188/3.86 (more faithful).

### Corrections to my own numbers

- Match-Query base: 0.888 +/- 0.140 (n=3) -> **0.730 +/- 0.247 (n=5)**.
- "No OOD degradation" was measured over 256->512 only. To 2048: 0.904 -> 0.693.
  Correct claim is "degrades gracefully", not "flat".
- Paper reproduction target: CLAUDE.md cited "0.955/0.999", which is in no table
  of the paper. Real Table 2 2D: MapWM 0.99/0.99/0.96, MapEM-os 1.0/0.99/0.97.

### Standing rules, each bought by a failure

1. n-gram on the ACTION STREAM ALONE at orders 1-5 before any demonstration task.
2. Context-destruction ablation on trained models.
3. Never compare a fresh variant to a stored baseline.
4. Report the measured chance rate beside every headline.
5. Verify the budget before reading a chance-level table as a negative.
6. **Three seeds is not a point estimate.**

### New infrastructure

`environment_match_query.py` + `validate_match_query.py` + `train_match_query.py`
+ `eval_match_longq.py` (the verified task, parameterised by size/n_obs);
`environment_lap.py` + `validate_lap.py` + `train_lap.py` + `probe_lap_theta.py`
(CSCG lap port, gated); `audit_planner_tasks.py` (the audit that voided four
tasks); `run_lap_transfer.py` (sequential-transfer harness with shared vocab).

### Late additions to 2026-08-09 (after the summary above)

**Timing benchmark** (`TIMING_BENCHMARK.md`) — first wall-clock measurement of the
parallel-scan claim here. Forward+backward, L=128 -> 2048:
parallel 2.6-3.3x, MapEM-NC 14.5x, TEMFaithful 120.2x. At L=2048 Vanilla is 34x
faster than MapEM-NC and 1632x faster than TEMFaithful (which has 20x FEWER
params). Qualifications in the file: parallel models are overhead-dominated below
L=1024; TEM's constant includes Python-loop overhead (the SHAPE is architectural).

**Family tree** (`FAMILY_TREE_RESULTS.md`) — built `MapEM-NC` (paper B.2.2:
K=n(n-1)/2 skew generators, sequential product) and the family-tree task the paper
motivates it with but never runs. On a structure with non-commutativity 1.000:
NC-NL 0.729, NC-L 0.720, COMMUTATIVE control 0.715, index 0.600. Non-commutativity
buys +0.005 to +0.014 for 34x the cost; path integration buys +0.115.
Floor is the hub baseline 0.146-0.163 (depth-dependent), NOT chance 0.125.

**EM on Match-Query** (`MATCH_QUERY_EM.md`) — single-p_0 beats paper-faithful
separate q0/k0 by **+0.358** (3/3 seeds; separate-form seed 0 collapses to 0.107).
Effect grows with reliance on A_P: paper task +0.089, compositional +0.167,
Match-Query +0.358. WM control reproduces the sweep's 0.888 to 3 d.p. EM still
never beats WM (0.808 vs 0.888). Mechanism deliberately NOT proposed -- the A_P
kernel-geometry account was falsified on a pre-registered test.

### Repo hygiene

47 void files moved to `archive/void/` with a README naming the three causes
(lm200 non-convergence, hier-goal action-prefix shortcut, planner-demonstration
shortcut). The DIAGNOSTICS that established each verdict are kept at top level --
they are current results. `RESULTS_INDEX.md` regenerated; it leads with the four
citable results and the seven standing rules.

### Harness traps that cost real time (2026-08-19)

Three ways to lose work that have nothing to do with the science:

1. **Anything over ~2 minutes must run under `setsid`, evaluators included.**
   A foreground tool call is SIGTERM'd at the 2-minute timeout and the signal
   propagates to every child it spawned. This happened THREE times in one
   session -- the family-tree extra arms, the knob-sweep evaluation, and the
   n=8 MiniGrid evaluation -- always because a job "looked short". Training
   pipelines were safe only because they were written as scripts. Write the
   evaluator as a script too, or it dies.
2. **`local a=$1 b=$2 c="...$b"` expands every word before assigning**, under
   `set -u`. Reference `$2` directly.
3. **`pgrep -f` / `pkill -f` match their own shell.** Split the pattern
   (`P="run_thing"".sh"`) and filter by `ps -o user=` before signalling --
   killing another user's processes was attempted once by accident.

Also: **do not edit a module while a batch is spawning processes from it.**
Later runs pick up the new code and a within-batch comparison silently becomes a
between-code one. If it is unavoidable, verify the affected configs are
unchanged (md5 the token stream) rather than assuming.

### Rule 7, added this session

**A gate must CALL the task code, not reimplement it.** `validate_family_tree.py`
originally duplicated the walk inline, so a dedup fix to the environment left
every gate number unchanged -- the gate was certifying a different task from the
one the trainer would run. This is nastier than the other failure modes because
the gate keeps looking like it works.


## Session 2026-08-19/20 — the headline changed: the ENVIRONMENT decides

**Read `BASELINE_TABLE.md` first.** It is the deliverable: every model x every
task, nine sections, each stating its own measured floor, seed count, and whether
its arms were trained in one batch. 37 commits this session, all pushed.

### The result the project now has

**No architectural ingredient is best. Which one to spend on is decided by the
environment, and one environment property decides most of it.**

The same 2x2 ({RoPE, PoPE} encoding x {index, path-integrated} position), two
environment families, each factor averaged over the other:

|                          | encoding | hierarchy | position |
|--------------------------|----------|-----------|----------|
| torus paper task (n=8)   | +0.011   | --        | **+0.461** |
| MiniGrid DK-16x16 (n=8)  | **+0.076** | +0.048  | **-0.021** |

Position is worth 40x the encoding on the torus and is NEGATIVE on MiniGrid.

> **CORRECTED 2026-08-30 -- the ALIASING explanation of this table is FALSIFIED,
> with the sign INVERTED.** The reading that survived the 2026-08-26 convergence
> work was "the position effect scales with observation ALIASING" (grid 8, 2
> cells/token, -0.010; grid 32, 32/token, +0.173; torus, 128/token, +0.461).
> That was correlational -- aliasing co-varied with map size across those
> environments. Holding grid size FIXED at 32 and varying n_obs alone gives the
> OPPOSITE ordering: **+0.178 (32 cells/token) -> +0.310 (8) -> +0.374 (2)** at
> 400 ep, and at full convergence the endpoints are **+0.178 vs +0.305**
> (t=2.52, both arms flat 10/10 and 6/6). LESS aliasing gives a LARGER position
> effect. Details in ALIASING_CONTROLLED.md; do not cite the aliasing story.

**Why: rotation-based actions.** Of the five properties differing between the
environments, rotate accounts for **-0.388 of the -0.438** swing (n=8), more than
the other four combined. MapFormer cumsums a FIXED per-token delta; under
turn/turn/forward the displacement depends on accumulated heading, which that
form cannot represent.

**The fix: allocentric action recoding.** Record the absolute displacement
instead of the commanded turn/forward -- dynamics byte-identical, gates
identical. Position effect +0.050 -> **+0.488** at 4 headings (8/8 seeds,
+/-0.005), exceeding the translate baseline. It also holds at Habitat's 12
headings (+0.26 to +0.38, present at every budget: the weakest path-integrated
seed 0.661 vs the strongest index seed 0.555 and a 0.508 floor). Needs no
architecture change and works wherever the agent's heading is known -- i.e.
every simulator.

CORRECTED 2026-08-23: this entry previously read "recovers once the budget is
adequate (+0.264 at 980 -> +0.383 at 2000, spread collapsing)". The nb=4000
point falsifies the trend -- it goes back DOWN to +0.286, with two of three
seeds converging worse (loss 0.834/0.815) than every seed at nb=2000
(0.507-0.552) and the third better than any run in the table (0.422). Bimodal
basin selection, not a dose-response curve. Accuracy correlates with final
training loss at r = -0.996 across all 18 runs, so the arm-to-arm variation is
convergence, not evaluation. nb=2000 is the best point measured. Separately, the
INDEX arm leaves the floor at nb=4000 (0.542-0.555, loss falling monotonically
1.73 -> 1.69 -> 1.59), which shrinks the measured effect on its own. Full
per-seed table in `H12_BUDGET_CURVE.md`.

### Other results that landed

- **MiniGrid 8-cell factorial, n=8** (`MINIGRID_FULL_2X2X2.md`): the two best
  arms are INDEX models (PoPE-Hier 0.955, PoPE-Flat 0.953); the paper's own
  MapFormer-WM is LAST (0.823). Hierarchy helps in inverse proportion to base
  strength (+0.096 for the weakest arm, +0.002 for the strongest) -- compensation,
  not addition. 27/32 paired positive; an earlier n=3 "18/18" was luck.
- **Level 1.5 across five within-batch tests**: improves LIKELIHOOD and
  STABILITY reliably, ACCURACY almost never. On lm200 a filter-free capacity
  control (Vanilla_ExtraHead) TIES it (t=0.79), so the +24.8pp is not evidence
  for the Kalman mechanism. lm200 passes its context-destruction gate but its
  interpretation is withdrawn.
- **Family tree**: the missing plain-WM arm beats every published variant
  (0.805 vs MapEM-NC-NL 0.729). Non-commutativity still buys +0.014 for 34x the
  cost -- but below plain MapWM-Flat.
- **Frequency control**: path-integrated arms learn omega, index arms do not, so
  every "position effect" was confounded. Measured: the confound is real in the
  code and ABSENT in the data (+0.004/-0.008).
- **Attention horizon is CAPACITY, not architecture** (`HORIZON_RESULTS.md`):
  ~2 steps at 1 layer, ~32 at 4 layers x d256 -- but a 15x larger index model
  still fails past ~32 where a 1-layer path-integrated model works at 65+.
- **Audit** (`AUDIT_HEADLINE.md`): all 23 headline numbers re-derived from
  per-seed JSON, 0 mismatches. Note what it does NOT establish -- every file in
  archive/void/ would have passed it.

### Habitat: BUILT AND VERIFIED, but NOT worth porting in this framing

`HABITAT_BUILD.md` has the full build log. habitat-sim 0.3.3 headless is
installed in a separate `habitat` conda env (py3.9 only -- it CANNOT share the
py3.12 + torch main env). Scenes in `habitat_data/` (gitignored, one command to
re-fetch). Eight unit tests pass: turns are exactly 30 deg, forward exactly
0.25 m, displacement depends on accumulated heading, 12x30 closes the circle.

**Three reasons not to port**, all found by building it and reading the field:
1. Habitat's navmesh SLIDES the agent on **69-91% of forward moves**, so real
   displacement is continuous in MAGNITUDE, not just direction. No experiment
   here models that.
2. Published Habitat numbers come from **RL-trained recurrent policies**
   (habitat-lab: ResNet18+GRU+DD-PPO; OVRL-V2 SOTA is ViT+**LSTM**). We do
   supervised next-token prediction. Not comparable without adopting DD-PPO.
3. ENTL (arXiv:2304.02639), the closest precedent, uses VQ-GAN at **256 image
   tokens per frame** -- a 500-step episode is ~130k tokens against this repo's
   4096 maximum.
Transformers ARE used on Habitat (ENTL, Scene Memory Transformer, Memo); the
reason recurrence dominates is PPO compatibility and sequence length, not
quality. **MiniWorld is the cheap 3D rung** if a third environment is wanted --
installed and verified headless already.

### Rules bought this session

- **Rule 5 caught THREE false negatives in one day**: rotate (+0.004 -> +0.050),
  Level15 on the paper task (0.938 at 16 epochs -> 1.000 at 50), and H=12
  allocentric (+0.264 -> +0.383). A weak number at one fixed budget is not a
  result. High seed variance is the tell.
  **But rule 5 cuts both ways, learned 2026-08-23:** two budget points make a
  line and a line is not a trend. Extending H=12 to nb=4000 sent the effect back
  DOWN (+0.383 -> +0.286). The correct move on seeing a budget-sensitive number
  is to report the per-seed spread against final training loss, not to fit a
  direction through two points.
- **Gate BEFORE training, not after.** The knob sweep trained 42 models and
  gated afterwards; the rotate condition was void (0.932 order-1 shortcut).
- **Gates must also check token ids are in vocabulary.** All three continuous
  conditions passed the answer-stream gates while one could never train -- an
  out-of-range embedding lookup surfaces as CUBLAS_STATUS_ALLOC_FAILED and reads
  exactly like CUDA OOM.
- **`wait` returns regardless of child success.** A script touched its .done
  marker after every arm died. Verify the artifact exists; do not infer it from
  the absence of a crash.
- **Flat-vs-hierarchical needs n_layers=3.** Hourglass variants IGNORE
  --n-layers and are always the 3-block scaffold, so at n_layers=1 it is 614K vs
  218K -- a 2.8x capacity confound.

### In flight at session end -- RESOLVED 2026-08-23

`run_h12_budget.sh` nb=4000 finished; `eval_h12_budget.py` +
`eval_h12_perseed.py` read it. It did NOT confirm: see the CORRECTED note under
"The fix: allocentric action recoding" above and `H12_BUDGET_CURVE.md`. The
DIRECTION (allocentric recoding works at 12 headings) survives at every budget;
the monotone budget curve does not. Open: where the nb=4000 bimodality comes
from -- same LR schedule shape and 2x fresh data, more steps at high LR is the
untested suspect. n=3 cannot separate "two unlucky seeds" from "the 4000-batch
recipe is worse"; more seeds at nb=4000 is the honest next step if the number
goes in a paper.


## Session 2026-08-29/30 -- the aliasing story dies; recursion; two retractions

Five things landed. Read ALIASING_CONTROLLED.md and LOOPED_PILOT.md for the data.

### 1. The aliasing claim is FALSIFIED and the sign is INVERTED

Grid size pinned at 32, only n_obs varied. The gates confirm this changes NOTHING
but the labelling: G5 label mass (50.4 scored/traj) and G6 revisit lag (median 33)
are byte-identical across all three conditions -- same walks, same scored
positions.

| n_obs | cells/token | effect | n | converged |
|---|---|---|---|---|
| 16 | 32 | +0.178 | 5 | 10/10 flat, 400 ep |
| 64 | 8 | +0.310 | 4 | 6/8 flat, 400 ep |
| 256 | 2 | **+0.305** | 3 | **6/6 flat, 800 ep** |

Monotone the WRONG WAY for the aliasing hypothesis, which pre-registered a
collapse below the 0.150 noise floor at n_obs=256. Endpoints differ by +0.127
(t=2.52) with every arm converged.

**The budget mattered and my convergence-sensitivity check pointed the wrong way.**
At 400 ep the n_obs=256 index arm was flat in only 1/5 seeds and the effect read
+0.374; the both-flat-only sensitivity said non-convergence was SUPPRESSING the
effect. Doubling to 800 ep converged it 3/3 and the effect fell to +0.305. The
sensitivity check was wrong; only the extension settled it. Trust budget
extensions over convergence-conditioning arguments.

**Repro control was exact**: RoPE n_obs=16 s0 retrained 0.725 vs stored 0.725,
drift +0.000. The pipeline is bit-reproducible across batches.

### 2. Map size, at MATCHED aliasing, is a THRESHOLD not a gradient

grid 8 @ n_obs=16 and grid 32 @ n_obs=256 both have 2.0 cells/token. Adding grid
16 @ n_obs=64 gives a three-point axis at constant aliasing:

| grid | occupied cells | effect |
|---|---|---|
| 8 | 32 | -0.010 |
| 16 | 128 | +0.015 |
| 32 | 512 | **+0.305** |

Flat, flat, JUMP -- somewhere between 128 and 512 occupied cells. NOT graded.
(My pre-registered "between" band was written too wide and mechanically fired
"graded"; +0.015 vs -0.010 is 0.025 against a 0.150 floor, i.e. identical.)

### 3. Recursion SUBSTITUTES for depth -- in the arm that needs depth

`model_looped.py`: one shared block applied 4x, param-parity exact with 1 layer
(207,457 vs L4's 802,273). Torus, 300 ep warmup+cosine, n=3.

- **INDEX arm: +0.363 at interval 17-32 (sd 0.018, MDE 0.029, 3/3 seeds)**,
  horizon 9-16 -> 17-32, statistically indistinguishable from 4 REAL layers
  (delta -0.023) at a quarter of the parameters. What depth bought was effective
  ITERATION, not layer specialisation.
- **PATH-INTEGRATED arm on the TORUS: no established gain**, and uninterpretable.
  +0.046, sd 0.074, MDE 0.120, one seed NEGATIVE. MapFormer already scores 0.948
  there with 0.052 of headroom; a ceiling cannot separate "adds nothing" from
  "nothing to add". **RESOLVED on Match-Query, where there IS headroom -- see the
  section below.**
- **The wall does not move.** Every index config still collapses past ~32
  (0.498/0.515/0.497) while path integration holds 0.945 at 65+ from ONE 204K
  layer. Recursion substitutes for depth and remains no substitute for path
  integration.
- Recursion buys PARAMETERS, not compute: four passes cost four passes.

### 4. RETRACTED: "scale HURTS the path-integrated model at long range"

HORIZON_RESULTS.md reported Vanilla non-monotone in capacity (L2 d256 0.976 ->
L4 d256 0.782 at interval 65+). Under warmup+cosine at 300 ep it does not
reproduce: L1 0.948, L4 0.998, Looped 0.994. It was an artifact of the 16-epoch
LinearLR budget, as its own caveat allowed. The whole published horizon TABLE is
budget-limited the same way -- RoPE L1's horizon is 9-16 under a fair budget, not
the ~2 originally reported or the ~8 of the 50-epoch point.

### 5. FALSIFIED: "distinct cells visited" (a hypothesis of mine, killed in 6 runs)

After the grid-16 threshold I proposed that what matters is how many distinct
cells the agent actually visits. Measured prior-visit counts first
(probe_visits_per_cell.py) -- my arithmetic said 16/4/1 for grid 8/16/32, the
truth is 8.64/4.61/3.05, because the walk is directed. That measurement also
showed the three original conditions are FULLY CONFOUNDED (distinct cells, prior
visits and map extent all move together), so none of them could test anything.

Condition A (grid 32, T=128: 48 distinct, 1.95 prior, 512 occupied) gives
**+0.275** where grid 8, T=512 (46 distinct, 8.64 prior, 32 occupied) gives
-0.010. Matched on distinct cells, opposite results -> distinct-cells-visited is
dead. Visits-per-cell and map extent both survive.

**RESOLVED 2026-08-30 by pooling: MAP EXTENT drives it, visits-per-cell does NOT.**
Condition B (grid 16, T=1024) landed at +0.010 -- but it is a CEILING condition, not
an informative null: the index arm SOLVES it (0.988 vs path integration's 0.995), so
153 distinct cells and 6.20 prior visits are no obstacle when the map holds 128
occupied cells. With two episode lengths now at each of two map sizes, prior visits
can be varied AT FIXED map extent:

| grid | occupied | T | prior | effect |
|---|---|---|---|---|
| 8 | 32 | 512 | 8.64 | -0.010 |
| 16 | 128 | 512 | 4.61 | +0.015 |
| 16 | 128 | 1024 | 6.20 | +0.010 |
| 32 | 512 | 128 | 1.95 | **+0.275** |
| 32 | 512 | 512 | 3.05 | **+0.305** |

Within a map size, prior visits move 1.34x / 1.56x and the effect moves 0.005 /
0.030 -- both far under the 0.150 floor. Across map sizes it moves **0.285**. The
decisive contrast: grid 16 at prior 4.61 gives +0.015 while grid 32 at prior 3.05
gives +0.305. A 1.5x change in prior ACROSS a map boundary flips everything; a 1.6x
change WITHIN a map does nothing. (Pooling is post-hoc; the pre-registered pairwise
test was more conservative and called it unseparated. Both are in VISITS_TEST.md.)

Limit: prior visits were varied only 1.3-1.6x at fixed map extent against a 4.4x
total range, and the two cannot be crossed here --

**Structural obstacle, worth knowing before designing anything here:** prior-visit
ranges by grid size DO NOT OVERLAP (grid 8 spans 5.67-18.35 over T=128..2048;
grid 32 spans 1.95-4.13). Small maps FORCE frequent revisits. Map extent and
visit statistics are near-inseparable in MiniWorld at any episode length. The one
condition that separates them is grid 32 @ T=2048 (prior 4.13, matched to grid 16
@ T=512's 4.61 which gave +0.015, on a 4x larger map). Not run -- seq 4096.

### Tooling that came out of this

- **`--fast-attn`** (model.py `USE_SDPA`, opt-in, default off): SDPA + TF32,
  **2.56x faster at 37% of the memory**, verified equivalent (logits 1.4e-06,
  gradients 2.4e-08, grad cosine 1.0000000000). Licensed empirically too: the
  control arm reproduced +0.392 against a +0.374 reference. NOT valid for MapEM
  (Hadamard A_X (*) A_P cannot be expressed as SDPA).
- **`--schedule cosine`** added to `train.py` and `train_match_query.py`; BOTH
  used LinearLR-from-step-one, like `train_miniworld` did. Default unchanged.
- **`prebuild_buffers.py`**: MiniWorld makes an EGL context per worker (~150 MiB
  of GPU) even with rendering disabled, so 6 jobs x 24 workers pinned a 4090 at
  97.5% before any model loaded, and oversubscribed 32 cores 4.5x. Prebuilt
  serially it peaks at 4.5 GiB, and runs went 5.5 h -> 68 min because the old
  batches were mostly building buffers, not training.

### Rules bought this session

13. **A fill-first GPU picker silently becomes a single-GPU scheduler** whenever
    the job count is <= MAXPG. Raising MAXPG 3 -> 6 idled a whole 4090 for 3 h.
    Balance to the LESS LOADED device. And do not then interleave job types --
    alternating types against an alternating picker phase-locks and puts every
    long job on one device (I did exactly this and reproduced the bug I was fixing).
14. **Do not infer held-out accuracy from training loss.** I predicted condition A
    would be null from RoPE's 0.03 training loss; it scored 0.674 held-out against
    Vanilla's 0.949. The r=-0.996 affine relation does NOT hold in every regime.
15. **A wide pre-registered band is not a pre-registration.** My grid-16 branch
    fired "graded" on +0.015 because the "between -0.010 and +0.374" band swallowed
    the competing "near zero" case. Set branch boundaries against the NOISE FLOOR.
16. **Editing a running bash script is unsafe** -- bash reads by byte offset and an
    insert can make it resume mid-token. Kill and relaunch instead (cheap when the
    script is parked in a wait loop).


## Loop x path integration on Match-Query (2026-08-31) -- they DO compose

`LOOP_HEADROOM.md`. The torus null above was a ceiling artifact. Match-Query 128^2
leaves real headroom, and all five arms were run to n=8 in one batch (300 ep,
warmup+cosine, fast-attn, chance 0.0625).

| arm | params | mean | sd | min |
|---|---|---|---|---|
| index, no loop | 204,182 | 0.108 | 0.025 | 0.06 |
| index + loop x4 | 204,182 | 0.207 | 0.032 | 0.15 |
| path-int, 1 layer | 204,630 | 0.456 | 0.220 | 0.11 |
| **path-int + loop x4** | **204,630** | **0.870** | **0.099** | **0.77** |
| path-int, 3 REAL layers | 601,174 | 0.771 | 0.263 | 0.14 |

- **Q2 loop on path integration: +0.414** (sd 0.279, MDE 0.277, 7/8) DETECTABLE
- **Q3 loop on index: +0.099** (sd 0.045, MDE 0.045, 8/8) DETECTABLE
- **2x2 interaction: +0.315** (MDE 0.281) DETECTABLE -- SUPER-ADDITIVE
- Q4 loop vs 3 real layers: +0.099, MDE 0.252, 5/8 -- UNDERPOWERED

**Both ingredients help and they compose super-additively.** This is the one place
in the project where stacking a mechanism onto MapFormer measurably pays. What was
missing on the torus was HEADROOM, not a different mechanism.

**RETRACTED from the n=3 read: "the loop BEATS three real layers by +0.273".** At
n=3 PI_L3 was 0.836/0.812/0.143; at n=8 the failure rate is 1/8, its mean is 0.771,
and Q4 is underpowered. **The loop MATCHES real depth at a third of the parameters,
it does not beat it.** (At n=2 I had read L3's 0.824 as "reproduces the published
0.823" -- three successive claims off n<=3 in one session, each dissolved by more
seeds. Rule 6 is about MY fresh numbers, not just other people's.)

**The most robust part is STABILITY.** The loop arm never fails: 8/8 seeds >= 0.77,
sd 0.099, against 1-layer's 0.11-0.80 (sd 0.220) and three real layers' 0.14-1.00
(sd 0.263). The loop's contribution is mostly to the FLOOR -- it makes an unreliable
model reliable at constant parameters rather than raising the best case. The one
seed where the 1-layer baseline trained well (0.800) is the one seed where the loop
did not help (-0.029). That is also why Q2 clears its MDE only narrowly despite a
large mean: the variance being removed is the baseline's, so paired differences
inherit it.

Scope: one task, one loop count (4), plain ALBERT-style sharing -- no per-iteration
depth embedding, theta computed once. The refine-theta-per-iteration variant
(iterative position refinement, structurally the InEKF work on the depth axis) is
untested and is now the natural follow-on.


## Filter x loop: NOT complementary (2026-09-01, n=12)

`L15_LOOP_2X2.md`. The 2x2 that had never been run -- there was no arm with both
until `Level15Looped` (verified bit-identical to Level15 at n_loops=1, causal leak
0, loop free on both rows, filter exactly 49,600 on both, so the INTERACTION is
parameter-matched). Clean torus, 5 arms x 12 seeds, one batch, 300 ep cosine.

| arm | T=128 | T=512 | T=1024 |
|---|---|---|---|
| Vanilla | 0.947 | 0.876 | 0.749 |
| Level15 | 0.990 | **0.953** | **0.878** |
| Looped | 0.999 | 0.872 | 0.730 |
| Level15Looped | 0.994 | 0.929 | 0.830 |
| LoopedSampled | 0.997 | 0.905 | 0.745 |

**Interaction UNMEASURED at every length** (T=512 +0.026 loss-matched vs MDE 0.099;
T=1024 +0.022 vs MDE 0.147; raw NEGATIVE at both). And the levels settle it without
needing the interaction: **the combination is WORSE than the filter alone at OOD**
(0.830 vs 0.878 at T=1024, 0.929 vs 0.953 at T=512). The best arm at OOD length is
Level15 by itself. The anti-correlated-profiles argument was suggestive and wrong;
the mechanism objection written before the run was right -- the loop's OOD damage
is iteration count, and bounding theta has no purchase on it.

**The loop's training-length win is CONVERGENCE, not representation.** Raw
Looped - Vanilla at T=128 is +0.052 (t 3.03, 12/12) but **loss-matched +0.006**,
and r(loss, acc) = -0.956 there. Mean final loss: Looped 0.0076, Level15Looped
0.0189, LoopedSampled 0.0180, Level15 0.0420, **Vanilla 0.1549**. The loop's real
contribution is that it converges reliably -- consistent with LOOP_HEADROOM's
"the loop's contribution is mostly to the FLOOR".

**Second instance of rule 14, and a sharp one.** r(final loss, accuracy) here is
**-0.956 / -0.471 / -0.326** at T=128/512/1024, where the L15 ablation had
-0.930/-0.897/-0.812. The coupling COLLAPSES with length in this arm set because
the loop arms all converge to ~0.01 loss but vary hugely OOD (Looped sd 0.166 at
T=1024). So the loop's OOD failure is NOT a convergence failure, and loss-matching
is well justified at training length and much less so at OOD. Check r per length
before leaning on a loss-matched residual.

The filter main effect REPLICATES the ablation: raw +0.129 (t 3.51) at T=1024,
loss-matched +0.083 (t 2.79) -- just under the t>2.8 bar at n=12, so still not
formally detectable, but the same size and sign as the ablation's +0.124 (t 3.83).

### Infra note bought the same day

A supplementary scheduler was added mid-batch to raise concurrency (data
generation is 79-95% of an epoch and single-threaded, so 24 of 32 cores sat idle).
It duplicated 6 of 66 launches. **A two-sided guard needs BOTH sides**: the new
script checked "checkpoint exists" AND "this variant+seed is already in flight",
but the ALREADY-RUNNING script could only check the first, so it relaunched
whatever was in flight without a checkpoint yet. Harmless here (same seed, same
code, and the two independent evals agreed byte-for-byte) but it cost ~10% of the
compute, and the second launch TRUNCATES the first's log, which is where final
training loss for the rule-9 analysis comes from.

Fix for next time, built and gated: `data_parallel.py` + `verify_data_parallel.py`,
opt-in `--data-workers` (2.15x at 3 workers, 3.43x at 6). Batch i is seeded by its
INDEX so the stream is invariant to worker count, but it DIFFERS from the serial
path's stream -- reproducible among themselves, never against a stored serial
checkpoint. Note the ceiling: for a saturated multi-job batch the GPUs are already
at 97-100%, so the win there is ~1.6x, not 3.4x. 3.4x is the single-run number.

## Session 2026-08-31/09-01 -- refinement is dead; the loop is the surprise

### IN FLIGHT at the time of writing

`run_l15_ablation.sh` -- 6 arms x 5 seeds on the clean torus paper task, 300 ep
warmup+cosine, results to `L15_ABLATION.md` via `eval_noise_refine`. It replicates
the single-seed Level 1.5 decomposition (below) at n=5. Started 12:19, ~3.5 h.
Marker `.l15_ablation_done`. Nothing is queued behind it.

### 1. REFINING THETA DOES NOTHING -- tested in the regime built for it

`NOISE_REFINE.md`. `LoopedRefine` carries and corrects the position estimate each
pass (`theta = theta_0 + gate * tanh(refine(x))`), i.e. the InEKF idea moved to the
DEPTH axis. Torus paper task under action noise, 4 arms x 3 noise x 3 seeds.

refine minus fixed-theta: **-0.001 / -0.011 / +0.005** at T=128 and
**+0.006 / +0.003 / -0.005** at T=512 for p_action_noise 0 / 0.10 / 0.25. Every
|t| < 2, and NO SLOPE in noise -- the pre-registered prediction was a positive
slope. The learned gate settles at mean|g| **0.083** with INCONSISTENT SIGN across
seeds, capping the correction at 0.14 rad against a theta spanning ~2*pi*T. The
gate was verified escapable before launch (gradient 1.9e-03 at zero), so the
optimiser DECLINED to refine. That is a mechanism answer, not a failure to express.

**A DESIGN ERROR THIS CORRECTED.** The first refine test was on Match-Query, where
actions are CLEAN and the query phase is BLIND -- neither half of the InEKF premise
holds. This repo had ALREADY measured that for the sequence axis ("Match-Query
(blind) 0.876 vs 0.888, nothing to correct with"), so that null replicated a known
negative. Check a mechanism's PREMISE applies before testing it.

### 2. THE CONTROL ARM WON: looping beats the Kalman correction under noise

Same batch, at TRAINING length, vs Vanilla:

| p_action_noise | loop (no filter) | Level15 (the InEKF) |
|---|---|---|
| 0.10 | **+0.138** (t=12.1) | +0.023 (t=1.9) |
| 0.25 | **+0.205** (t=8.8) | +0.004 (t=0.2) |

A shared block applied four times is ~9x more effective under action noise than
the purpose-built correction, at FEWER parameters (204K vs 254K). n=3, and it came
from reading the control column rather than from a prediction -- needs its own
pre-registered replication before it is a claim.

**Level15's only detectable effect is at OOD LENGTH** (+0.025, t=3.82 at T=512
p=0.25; +0.004 at T=128). That is the signature of stabilisation, not inference.

### 3. THE LOOP'S COST IS LENGTH -- and it is mostly trainable away

Degradation T=128 -> T=512, averaged over noise: Level15 -0.055, Vanilla -0.065,
**Looped -0.243, LoopedRefine -0.240**. Looping degrades ~4x worse with length.

I first attributed this to residual-scale growth by analogy to the wrap finding,
WITHOUT measuring. The residual norm is flat across length (18.15 -> 18.71), so
that mechanism is wrong. An EVAL-ONLY loop-count sweep -- free, since n_loops is a
runtime argument, where I had proposed 12 training runs -- shows the damage IS the
iteration count: same weights, T=512 peaks at **2** passes (0.794) and falls to
0.766 at 6, while T=128 rises to 1.000 at 4.

`LoopedSampled` (count drawn from {2..6} per training batch, param-identical)
then gives (`LOOP_SAMPLED.md`, n=5):
- **The count-vs-accuracy curve FLATTENS from 0.178 spread to 0.001.** The sampled
  model scores **0.998 at ONE pass** where the fixed-count model collapses to
  0.821 -- 4x cheaper inference for free. This is the tightest result of the set
  and was NOT the pre-registered question.
- OOD gain +0.092 (T=512) / +0.085 (T=1024) at its best count, but **t=1.67/1.80
  at n=5 -- directionally right, NOT established**. Needs ~n=12.
- **Does not transfer to noise**: +0.017 / +0.015 at OOD, -0.014 at train length.
- Even repaired, the loop does NOT beat Vanilla at T=1024 clean (0.736 vs 0.767).

### 4. What Level 1.5 is made of -- RESOLVED at n=5: it is made of NOTHING NAMEABLE

`L15_ABLATION.md` (6 arms x 5 seeds, clean torus, 300 ep warmup+cosine). The
single-seed decomposition below does NOT replicate.

- **RETRACTED: "removing the token-type gate is worse than doing nothing."** At
  n=1 ConstR 0.672 < NoCorr 0.833. At n=5 ConstR is one of the two BEST arms and
  beats NoCorr on 5/5 seeds. Sign inverted.
- **WITHDRAWN: "Level15 does not reduce to clamping theta."** Level15 - NoMeas is
  +0.015/+0.074/+0.073 at T=128/512/1024 against MDEs of 0.033/0.137/0.105 --
  unmeasured raw AND loss-matched (+0.036, t 1.23). Not established in either
  direction; the n=1 gap (0.831 vs 0.993) does not reproduce.
- **L15_DARE == Level15** at every length: the principled Kalman gain buys nothing
  a learned scalar does not. (Consistent with n=1, still underpowered.)
- **Rule 9 flips two readings.** r(final loss, acc) = -0.930/-0.897/-0.812 over the
  30 runs. Raw, the ONLY detectable contrast is ConstR > NoCorr -- and NoCorr is
  the worst-converging arm in the set (2/5 flat, mean loss 0.195). Loss-matched it
  vanishes (t 0.51), while **Level15 - Vanilla APPEARS**: +0.062 (t 3.08) at T=512
  and +0.124 (t 3.83) at T=1024, unmeasured at T=128. Raw, even the headline is
  inside its MDE.

So the filter's effect is real and is confined to OOD LENGTH -- the stabilisation
signature this project already reported -- but **no individual component can be
shown to be load-bearing**: measurement head, per-token gate and learned Pi can
each be removed alone at no measurable cost. Do not cite the named decomposition.

### 4b. The original single-seed table (superseded by the above; kept for the record)

| arm | T=128 / T=512 | keeps |
|---|---|---|
| Level15 | 1.000 / 0.993 | wrap + measurement + per-token R |
| L15_DARE | 1.000 / 0.992 | same, Pi fixed by DARE -> the principled gain is irrelevant |
| L15_NoMeas | 0.904 / 0.831 | **the wrap alone -- a pure bounded clamp** |
| L15_NoCorr | 0.940 / 0.833 | nothing (== vanilla) |
| L15_ConstR | 0.795 / 0.672 | wrap + measurement, NO gate -- WORSE THAN NOTHING |

So Level15 does NOT reduce to clamping theta (NoMeas 0.831 vs 0.993), and the
per-token gate is load-bearing. Neither piece is inference: the decomposition is
**bounded state + token-type gating**, wearing Kalman clothing. Clean config only
-- the lm200 column is under the 2026-07-16 retraction.

### 5. Language facts, for the record

enwik8 flat 9-layer, ~28.6M params, 36k iters, seq 512 (`enwik8_long/*.json`):

| arm | encoding | position | bpc | n |
|---|---|---|---|---|
| MapPoPE-Flat | PoPE | path-int | 1.3740 | 3 |
| PoPE-Flat | PoPE | index | 1.3746 | 1 |
| Vanilla (MapWM) | RoPE | path-int | 1.3758 | 1 |
| RoPE | RoPE | index | 1.3799 | 3 |

Only MapPoPE-RoPE has seeds on both sides: **-0.0058, t=3.49**. The two
single-axis arms are n=1.

**Our position effect (-0.0041) does NOT contradict the paper.** v4 sec 5.5 gives
RoPE 19.14 vs MapWM 18.79 ppl = 0.0266 bits/BPE-token ~= **0.0067 bits per BYTE**
at 4 bytes/token. Our seed sd is 0.0018, so MDE at n=3 is 0.0041 -- the size of
the effect. This setup is UNDERPOWERED, not null. I called it "inside the noise
floor" and that was a rule-11 violation on my own data.

Hint worth testing: PoPE alone -0.0053, path-int alone -0.0041, sum -0.0093, but
both together only -0.0058 -- on text the two axes OVERLAP (sub-additive), the
opposite of loop x path-integration on navigation. Both singles are n=1.

### 6. Two mechanism clarifications worth not re-deriving

- **theta on language.** `ActionToLieAlgebra` reads EVERY token identically and the
  model LEARNS Delta ~= 0 for observations (verified on a trained torus model:
  actions move position 5x more than observations). Standard RoPE is the
  **Delta == 1 special case** -- `angle = t * inv_freq` vs `angle = cumsum(Delta) * omega`.
  So on text theta is a learned content-dependent CLOCK RATE, and can be zero or
  negative. That is Selective RoPE's primitive.
- **PoPE's angle is NOT learned or content-dependent.** `magnitude = softplus(Q)`
  comes from content; the phase is `t * theta_c` with theta_c a FIXED buffer. The
  only learnable angle term is `pope_delta`, a per-(head,frequency) constant bias.
  MapFormer and PoPE modify ORTHOGONAL halves of the same polar decomposition,
  which is why MapPoPE is just "use both". PoPE is not in the CoPE/Selective-RoPE
  family despite the name.

### Rules bought this session

17. **Check a mechanism's PREMISE applies to the task before testing it.** The
    first refine-theta test had neither drift to correct nor observations to
    correct with, and the repo already said so.
18. **Before proposing a training sweep, check whether the knob is a RUNTIME
    argument.** Loop count is. The eval-only sweep cost 90 seconds where I had
    specified 12 training runs, and it is what actually found the mechanism.
19. **Split a hypothesis before testing it.** I asserted "iteration compounds the
    damage VIA residual growth" as one claim; the residual half was wrong and I
    retracted the whole thing, then the iteration half turned out to be right.
    State the parts separately so a failed test kills only what it hits.
20. **`train_hourglass_enwik8` saves metrics but NO checkpoints**, so no post-hoc
    diagnostic on a trained language model is possible without retraining. Worth
    fixing before the next language run.

