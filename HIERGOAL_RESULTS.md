# Hierarchical goal-directed navigation — the MapFormer × hierarchy synergy

**The positive result that closes the hierarchy arc.** Everything prior showed
the time-hierarchy helping *generically* (works on a plain transformer too;
task-aware segmentation / collapse didn't help or hurt), and MapFormer barely
beating plain — the two helping orthogonal metrics with no interaction. This
task was designed to create the one demand those lacked: **absolute position at
multiple scales at once, sustained past the training horizon.** Here the
combination is genuinely super-additive.

## Task (see `environment_hier_goal.py`, `validate_hier_goal.py`)

Episode `[room_goal, local_goal, explore·T_e, navigate·T_n]`. Fixed start anchor
(0,0) → path integration yields *absolute* position; hierarchical goal (coarse
room id + fine local id) → target cell needs both scales; explore forces path
integration; navigate = BFS-optimal actions; loss = next-action CE (chance 0.25,
BFS ceiling 1.00). Trained at T_explore=64; evaluated at 64 / 128 / 192 / 256
(>64 = OOD explore length). obs redrawn per episode so localisation must come
from path integration, not memorised obs.

## Result (n=3, held-out env, `HIERGOAL_MULTISEED.md`)

Held-out action accuracy, mean ± std:

| variant | T=64 | T=128 | T=192 | T=256 |
|---|---|---|---|---|
| **MapWM-Hier** | 0.963 | **0.907 ± 0.026** | **0.849 ± 0.065** | **0.853 ± 0.059** |
| MapWM-Flat | 0.958 | 0.656 ± 0.206 | 0.746 ± 0.179 | 0.727 ± 0.188 |
| Plain-Hier | 0.968 | 0.700 ± 0.138 | 0.682 ± 0.122 | 0.624 ± 0.104 |
| Plain-Flat | 0.966 | 0.548 ± 0.084 | 0.669 ± 0.106 | 0.591 ± 0.117 |

**In-distribution (T=64) all four tie at ~0.96.** The task only discriminates OOD.

## The synergy is a genuine 2×2 interaction

interaction = (hierarchy's help on MapFormer) − (hierarchy's help on plain):

| eval length | Δ hier on MapWM | Δ hier on plain | **interaction** |
|---|---|---|---|
| T=128 | +0.251 | +0.152 | **+0.099** |
| T=192 | +0.103 | +0.013 | **+0.090** |
| T=256 | +0.126 | +0.033 | **+0.093** |

A consistent **+0.09–0.10 positive interaction at every OOD length** — the whole
exceeds the sum of parts. Equivalently, MapFormer's edge over plain is +0.08–0.14
flat but **+0.17–0.23 within the hierarchy**: the two amplify each other. This is
exactly what the compositional task lacked (there the hierarchy helped both
backbones equally → zero interaction).

`MapWM-Hier` is also uniquely **stable** OOD (±0.026–0.065 vs ±0.08–0.21 for the
rest) — the only variant that generalises reliably. NLL agrees emphatically
(0.36 vs 0.86–1.6 at T=128).

## Mechanism (why here and not before)

The objective needs absolute position at room scale *and* cell scale, sustained
over a horizon longer than training. MapFormer's path-integration code
extrapolates to OOD length (its known strength); the hierarchy gives it a stable
coarse channel to extrapolate *with* — region-scale path integration is even more
length-robust than cell-scale. Plain index-RoPE has neither the length-robust
position nor a position-bearing coarse level, so it degrades OOD with or without
the hierarchy.

## Honest caveats

- **n=3, single task family; the effect lives entirely at OOD length** — in
  distribution the four are tied.
- **Correction to the single-seed scan:** the scan showed Plain-Hier *collapsing*
  (0.48) and I read it as "hierarchy hurts plain." That was seed noise — at n=3
  hierarchy helps plain slightly too (+0.15/+0.01/+0.03). The real story is a
  super-additive interaction (hierarchy helps both, MapFormer far more), NOT a
  sign flip.
- MapWM-Flat is high-variance OOD (±0.18–0.21); part of MapWM-Hier's win is that
  the hierarchy *stabilises* MapFormer's OOD generalisation, not only raises its
  mean.

## Bottom line

On hierarchical goal-directed navigation at OOD explore length, MapFormer and the
time-hierarchy are load-bearing **together**: `MapWM-Hier` is best by a wide,
reliable margin with a consistent positive interaction. This is the "true
combination" the compositional task could not exhibit — and it required the
*task* to create the multi-scale-position demand, not a cleverer architecture.

## Follow-up: coarse position need NOT be spatial (CoarseIdx) — a correction

`MapWM-Hier-CoarseIdx` (commit a7a0c5e): identical to `MapWM-Hier` but the coarse
layer's rotation uses the coarse token INDEX (`omega*index`) instead of the pooled
fine path angle — the fine spatial angle is not transmitted upward. Param-identical,
causal. Prediction was: ~= on compositional, worse on hier-goal. **Both wrong.**

- **Hier-goal (n=3):** CoarseIdx ~= MapWM-Hier (T=128 0.846 vs 0.907; T=192 0.862 vs
  0.849; T=256 0.845 vs 0.853), with LOWER variance, still far above plain. The
  synergy does NOT require the coarse position to be spatial.
- **Compositional (n=3):** CoarseIdx is DRAMATICALLY better — cross_nb 0.619 ± 0.016
  at T=256 vs MapWM-Hier 0.423 ± 0.144. Best variant on the task, and it kills the
  variance (the "one lucky seed" instability was the pooled-angle coarse position).

**Correction to the mechanism:** the pooled spatial angle at the coarse level was a
liability, not an asset — `cum_delta` is unbounded and its window-mean is noisy,
which destabilised training and (on the content task) interfered with content-based
motif matching. A clean ordinal coarse position is better on both tasks. The
synergy's real ingredients are FINE-level path integration + a hierarchy (any clean
coarse channel), NOT a spatial coarse map. This vindicates the original Hourglass
choice (index/relative position at coarse), not our pooled-angle deviation.
CoarseIdx is now the best variant on both tasks.

## The coarse-position 2x2 completed: CoarsePI resolves it (spatial vs ordinal is task-dependent)

`MapWM-Hier-CoarsePI` (commit 165784e): the coarse level runs its OWN path
integration (dedicated coarse ActionToLie over the pooled tokens, +384 params),
DISCONNECTED from the fine cum_delta -- an adaptive spatial coarse map without the
pooling noise. This fills the last cell of the 2x2:

| coarse position | connected (pooled) | disconnected (own) |
|---|---|---|
| path-angle (spatial) | MapWM-Hier | **CoarsePI** |
| index (ordinal)      | --          | CoarseIdx |

Results (n=3):
- **Hier-goal (position task): CoarsePI is BEST at every OOD length** (T=128 0.915,
  T=192 0.890, T=256 0.894), beating pooled MapWM-Hier (0.907/0.849/0.853) AND
  index CoarseIdx (0.846/0.862/0.845), with the lowest variance (±0.009-0.019).
- **Compositional (content task): CoarsePI ~= MapWM-Hier (0.451 vs 0.423), both
  far below CoarseIdx (0.619).** A clean spatial coarse position still loses to
  ordinal here.

**Resolved mechanism -- the right coarse position is TASK-DEPENDENT:**
- Position task -> SPATIAL wins (CoarsePI > CoarseIdx). Since CoarsePI > MapWM-Hier,
  the pooled version failed ONLY from noise; a clean spatial coarse map is best.
- Content task -> ORDINAL wins (CoarsePI << CoarseIdx). Spatial-ness itself
  interferes with content matching; cleaning the noise doesn't help.
So MapWM-Hier (pooled) was bad for TWO reasons: spatial (hurts content) AND noisy
(hurts everything). CoarsePI removes the noise; CoarseIdx removes the spatial-ness.

Bonus: CoarsePI is now the strongest synergy variant -- it raises the hier-goal
headline (0.894 vs 0.853 at T=256) and is rock-stable. MapFormer x hierarchy gets
stronger. (Idea due to the user: give the coarse level its own path integration.)

## PoPE largely SUBSUMES the synergy on OOD length (the big reframe)

Added PoPE (Gopalakrishnan et al. 2025) variants (commit 78b1a95): PoPE decouples
content (magnitude=softplus) from position (phase only), so content can't shift
the position tuning at long range. Three variants, param-identical to their RoPE
counterparts: PoPE-Flat (index+PoPE), MapPoPE-Flat (path-int+PoPE), MapPoPE-Hier.

**Hier-goal (n=3): PoPE gives FLAT length extrapolation and subsumes both path
integration and the hierarchy.** All three PoPE variants hold ~0.95 at EVERY
length (T=64..256) with near-zero variance (±0.000-0.001), vs the previous best
CoarsePI dropping 0.961->0.894 and MapWM-Flat collapsing to 0.727.

| variant | T=64 | T=128 | T=192 | T=256 |
|---|---|---|---|---|
| PoPE-Flat (index+PoPE) | 0.952 | 0.950 | 0.950 | 0.947 |
| MapPoPE-Flat | 0.952 | 0.950 | 0.951 | 0.948 |
| MapPoPE-Hier | 0.951 | 0.949 | 0.950 | 0.948 |
| CoarsePI (prev best) | 0.961 | 0.915 | 0.890 | 0.894 |
| MapWM-Hier | 0.963 | 0.907 | 0.849 | 0.853 |

Crucially **PoPE-Flat uses plain INDEX position -- no path integration, no
hierarchy -- and ties MapPoPE-Flat and MapPoPE-Hier.** So on OOD length, PoPE's
decoupling subsumes the path-integration advantage AND the hierarchy advantage.
The MapFormer x hierarchy synergy we found was fixing a SYMPTOM (RoPE's
content-position entanglement degrading attention at long range); PoPE fixes the
root cause and more completely. This reframes (and partly deflates) the synergy
headline on the length axis.

**Compositional (n=3): PoPE helps content but does NOT dominate.** Decoupling
reduces content-position interference -- MapPoPE-Flat 0.363 > MapWM-Flat 0.270,
MapPoPE-Hier 0.466 > MapWM-Hier 0.423 (T=256 cross_nb) -- exactly as predicted.
But CoarseIdx (0.619) still wins by a wide margin: for pure content matching, the
ordinal coarse position beats PoPE. MapPoPE-Hier (0.466) is the best PoPE variant
here, so the hierarchy still earns its keep on content.

**Synthesis:** OOD length -> PoPE wins outright (subsumes the synergy). Content
transfer -> PoPE helps (validates the decoupling thesis) but the ordinal-coarse
CoarseIdx is still best. No single mechanism dominates both axes; PoPE is the
principled fix for the length/entanglement axis specifically.

## Best-of-both (PoPE + index coarse) + clock transfer: two honest qualifications

`MapPoPE-Hier-CoarseIdx` (commit 322c27e) = PoPE decoupling everywhere + ordinal
INDEX coarse position + path-integration fine + hierarchy. Aimed to win BOTH the
length axis (PoPE) and the content axis (CoarseIdx). Plus a PoPE-on-clock scan.

**1. No single variant wins both axes (the synthesis failed).**
- Hier-goal: best-of-both keeps PoPE's flat extrapolation (0.948 at T=256, ties
  the other PoPE variants).
- Compositional: best-of-both = 0.452 +-0.115 (T=256), ~= MapPoPE-Hier (0.466) and
  FAR below CoarseIdx (RoPE+index, 0.619). Swapping RoPE->PoPE in the CoarseIdx
  architecture DROPPED content 0.619 -> 0.452. PoPE and the ordinal-coarse content
  win do NOT stack: PoPE's softplus-magnitude, phase-only attention (which buys
  length robustness) also CAPS the sharp content matching CoarseIdx exploits.
- So it's a genuine Pareto trade-off: PoPE end = great length / mediocre content;
  CoarseIdx end = great content / mediocre length. The best-of-both did not
  collapse the frontier.

**2. PoPE's flat length extrapolation is partly hier-goal-specific.** On the
symbolic modular-clock task (seed 0), ALL variants degrade OOD -- nobody
flat-extrapolates. PoPE/best-of-both lead at T=128 (0.77-0.83 vs MapWM 0.64-0.70)
but the edge shrinks/reverses by T=256 (MapWM-Hier 0.681 > best-of-both 0.626).
So the earlier "PoPE subsumes everything on length" was overstated -- it was
strong on hier-goal, only moderate and non-flat on the clock. (Single seed; the
non-flat pattern is clear but exact ranks are noisy.) See CLOCK_SCAN.md.

Net: the two-axis picture stands, but there is no one-variant-to-rule-them-all;
PoPE's length benefit is task-dependent, and it trades against content sharpness.

## Clock transfer — de-noised (n=3): confirms PoPE's length win is task-dependent

Re-ran the clock scan at n=3 (commit follows). The single-seed read holds and
sharpens:
- **PoPE does NOT flat-extrapolate on the clock** (all variants degrade ~0.98 -> ~0.6).
  The hier-goal flat-0.95 result does not transfer to the symbolic domain.
- **PoPE's advantage is real but TRANSIENT.** At T=128 the PoPE variants clearly
  lead (0.77-0.83 vs 0.60-0.73, non-overlapping bars). By T=256 it's gone --
  everyone clusters ~0.60-0.64 (PoPE-Flat 0.636 ~= MapWM-Hier 0.644).
- **Mild reversal:** on the clock MapWM-Hier is the most STABLE far-OOD variant
  (0.629->0.644, ±0.02) while PoPE variants decay (MapPoPE-Hier 0.831->0.596) --
  almost the opposite of hier-goal.

Net: the "PoPE wins length" headline is task-dependent at n=3 (not noise); its
magnitude ranges from flat-and-dominant (hier-goal) to a transient bump that
fades (clock). See CLOCK_SCAN.md.

## Faithful PoPE (with delta_c) PARTIALLY OVERTURNS the Pareto claim

The earlier PoPE runs used a PoPE-LITE layer that omitted the paper's learnable
per-frequency phase bias delta_c (Eq. 6). Adding it (commit 4d990d4) and
retraining the whole PoPE arm changes the content story:

Compositional cross_nb (T=256), PoPE-lite -> faithful:
- MapPoPE-Hier-CoarseIdx (best-of-both): 0.452 -> **0.549** (+0.097)
- MapPoPE-Hier: 0.466 -> 0.455 ; PoPE-Flat: 0.291 -> 0.319 ; MapPoPE-Flat: 0.363 -> 0.366

So delta_c specifically rescued the BEST-OF-BOTH variant, and the effect grows
with length:

| variant | T=256 | T=1024 | T=2048 |
|---|---|---|---|
| MapWM-Hier-CoarseIdx (RoPE, content king) | 0.619 ± 0.016 | 0.369 | 0.282 ± 0.108 |
| MapPoPE-Hier-CoarseIdx (faithful) | 0.549 ± 0.112 | 0.334 | **0.287 ± 0.148** |

Gap at T=256 narrowed 0.167 -> 0.070; by T=2048 they are TIED. And on hier-goal
the same variant KEEPS the flat length win (0.948 at T=256, unchanged).

**Correction:** the previous "PoPE's attention caps content sharpness, the
length/content frontier cannot be collapsed" conclusion was partly an artifact of
the incomplete (delta-less) implementation. With faithful PoPE the best-of-both is
flat on length AND near-parity on content.

**Honest caveats:** the best-of-both's content variance is large (±0.112 at T=256,
±0.148 at T=2048) vs CoarseIdx's ±0.016, so the defensible claim is "gap
substantially narrowed, plausibly closed at long T", NOT "solved". Our PoPE also
still uses d/2 frequencies rather than the paper's d. Clock is essentially
unchanged (MapPoPE-Hier 0.831 -> 0.847 at T=128); the transient-then-fade pattern
holds.

## n=8: the frontier IS collapsed -- and CoarseIdx's "stability" was a 3-seed artifact

Ran seeds 3-7 for the two frontier-deciding variants on the content axis (n=8).
Two findings, one of which corrects earlier claims in this file.

**1. The length<->content frontier is effectively collapsed.**

| variant | T=256 | T=1024 | T=2048 |
|---|---|---|---|
| MapWM-Hier-CoarseIdx (RoPE) | 0.556 ± 0.123 (n=8) | 0.308 | 0.232 |
| MapPoPE-Hier-CoarseIdx (best-of-both) | 0.528 ± 0.117 (n=8) | 0.302 | 0.243 |

Statistically tied: gap 0.028 vs ±0.12 spread, and PAIRED BY SEED the sign is a
coin flip (4/8 each way, diffs from -0.129 to +0.247). Since the best-of-both
also holds the flat hier-goal length win (0.948), it WEAKLY DOMINATES -- as good
on content, far better on length. The earlier "Pareto trade-off" framing does not
survive n=8.

**2. CORRECTION: CoarseIdx's low variance was a 3-seed artifact.** Per-seed
T=256: 0.638 / 0.621 / 0.599 (seeds 0-2, the original n=3) then 0.575 / 0.480 /
0.634 / **0.258** / 0.640. Seeds 0-2 happened to land within 0.04 -- luck. At n=8
it is ±0.123 with a catastrophic seed. So the repeated claims above that CoarseIdx
"kills the variance" and scores 0.619 are WRONG: true mean 0.556 ± 0.123.

**Scope of the correction.** The whole compositional/content axis is noisier than
reported. Every n=3 ranking on that task (MapWM-Hier 0.423, CoarsePI 0.451,
MapPoPE-Hier 0.455, ...) carries ±0.11-0.17 and should be treated as PROVISIONAL
until re-run at higher n. The hier-goal/length axis is unaffected (±0.001-0.02
there, and the PoPE flat-extrapolation result is robust).

## n=8 content axis: a third correction -- "ordinal beats spatial" does NOT survive

Topped up 6 compositional variants + the clock to n=8 (commit 2eb2e42). Paired
per-seed tests at T=256 (n=8, two-sided, normal approx -- indicative):

| comparison | mean diff | wins | p | verdict |
|---|---|---|---|---|
| CoarseIdx - MapWM-Hier | +0.140 | 6/8 | 0.02 | SIGNIFICANT |
| CoarsePI - MapWM-Hier | +0.083 | 6/8 | 0.12 | not sig |
| **CoarseIdx - CoarsePI** | **+0.057** | **5/8** | **0.43** | **NOT SIG** |
| MapPoPE-Hier - MapWM-Hier | +0.018 | 5/8 | 0.55 | tied |

**OVERTURNED: the content half of "coarse position is task-dependent".** The
claim that ORDINAL (CoarseIdx) beats SPATIAL (CoarsePI) for content rested on
n=3 values 0.619 vs 0.451 (gap 0.168). At n=8: 0.556 vs 0.498, p~0.43, 5/8 seeds
-- indistinguishable. The spatial-vs-ordinal distinction survives only on the
POSITION task (hier-goal, tight bars); on content it dissolves.

**The contested trio is unresolved.** CoarsePI (0.498 ± 0.109), MapPoPE-Hier
(0.429 ± 0.117), MapWM-Hier (0.415 ± 0.096) are statistically tied. The only
significant pairwise result among these is CoarseIdx > MapWM-Hier (the pooled
baseline).

**HOLDS at n=8: hierarchy helps content, in both families.**
- MapFormer: MapWM-Hier 0.415 vs MapWM-FlatHG 0.285 -> +0.130
- Plain: Plain-Hier 0.318 ± 0.029 vs Plain-Flat 0.216 ± 0.004 -> +0.102
This is now the most robust finding on the content axis.

**NEW: the variance is STRUCTURAL, split by family not by seed luck.** Plain
variants ±0.004-0.029 (tight); MapFormer variants ±0.067-0.134 (wide). MapFormer
is intrinsically unstable on the content task -- a finding in itself, not noise
to be averaged away.

**Clock at n=8:** MapWM-Flat rose to 0.790 at T=128 (from 0.733 at n=3), so
PoPE's lead there is now overlapping (0.80-0.82 vs 0.790) rather than clean; by
T=256 all variants tie ~0.62-0.65. PoPE's clock advantage is weaker than n=3
suggested.

**Pattern worth noting:** three separate n=3 content-axis claims have now
dissolved under more seeds (CoarseIdx's stability, PoPE's content cap, and
ordinal-beats-spatial). Only effects >= ~0.10 with tight-variance families have
survived. Treat any content-axis gap < 0.10 in this project as unresolved.

## MECHANISM CORRECTED: pooling averages the spatial signal AWAY (it is not "noisy")

`probe_coarse_angles.py` extracts the coarse rotation angle theta = omega*cum from
the trained checkpoints (seed 0, hier-goal, held-out env). Measured:

| coarse angle | \|theta\| final (T=64) | across-episode sigma | sigma/\|theta\| | OOD drift (T=256/T=64) |
|---|---|---|---|---|
| learned (CoarsePI) | 65.7 | 6.61 | **10.0%** | 2.25x |
| pooled (MapWM-Hier) | 51.1 | 3.13 | 6.1% | 2.52x |
| ordinal (CoarseIdx) | 195.1 | 0.00 | 0.0% | 2.50x |

Two corrections to what this file said earlier:

1. **"cum_delta is unbounded and its window-mean is noisy" was wrong on both counts.**
   ALL THREE designs drift ~2.3-2.5x beyond their trained range at T=256 -- including
   the deterministic ordinal one. Unbounded growth is therefore NOT what separates
   them, and the pooled angle is not noisier: its across-episode sigma (3.13) is
   LOWER than the learned one's (6.61).

2. **The real difference is how much PATH-DEPENDENT signal the angle carries.**
   across-episode sigma measures how much theta varies with where the agent actually
   went. Ordinal = 0.00 by construction (pure ordering, no space). Pooled = 3.13.
   Learned = 6.61, i.e. ~1.6x more signal per unit of drift (10.0% vs 6.1%).
   Mean-pooling over k tokens AVERAGES THE SPATIAL VARIATION AWAY -- it produces a
   washed-out, flatter spatial code, not a noisy one. That is why the learned coarse
   path integration wins the position task.

Visual: coarse_angles.json + the published angle figure.

## Dimensionality sweep: the SSP 3/sqrt(D) prediction FAILS

Motivated by the SSP/VSA correspondence (Komer et al. 2019): the discriminability
floor for random vectors is sigma=sqrt(1/D), 3-sigma = 3/sqrt(D). Our d_head=64
gives 0.375 vs the 0.133 of the d=512 that literature converged on. Prediction:
OOD degradation is partly under-dimensioning, and raising d should fix it.

Held-out accuracy at T=256 (OOD), n=3, hier-goal:

| variant | d=128 (floor .375) | d=256 (.265) | d=512 (.188) |
|---|---|---|---|
| MapWM-Flat | 0.727 | 0.743 | 0.768 |
| MapWM-Hier | 0.853 | 0.774 | 0.848 |
| Plain-Flat | 0.591 | 0.611 | 0.605 |
| Plain-Hier | 0.624 | 0.901 | 0.751 |

**PREDICTION NOT CONFIRMED.**
1. The OOD collapse persists at d=512: MapWM-Flat still falls 0.969 (T=64) ->
   0.768 (T=256); Plain-Flat 0.980 -> 0.605. Halving the crosstalk floor with 15x
   the parameters (634K -> 9.6M) did not rescue length generalisation.
2. What d buys is small and IN-DISTRIBUTION: ~+0.01-0.02 at T=64 for every
   variant including the plain controls -> generic capacity, not a position-code
   effect.
3. **The hierarchy advantage is NOT a low-d artifact** (the reframe risk):
   MapWM-Hier - MapWM-Flat at T=256 = +0.126 / +0.031 / +0.080 across
   d=128/256/512. Noisy and non-monotone but present at every d. The synergy
   headline survives.
4. Variance does NOT shrink with d (±0.05-0.19 throughout), reinforcing that the
   instability is structural rather than a dimensionality limit.

**Limits of this negative:** at n=3 with ±0.15 bars we can only resolve effects
> ~0.15. MapWM-Flat at T=128 shows 0.656 -> 0.832 (+0.176), the one trend large
enough to possibly be real, and it is the variant the theory most directly
targets. So: "no effect resolvable at n=3", not "no effect". Plain-Hier's d=256
outlier (0.901, best in the sweep, then 0.751 at d=512) shows how noisy this
surface remains.

**Import for the VSA correspondence:** the one actionable number the SSP
correspondence produced did not pay off. The correspondence stands as
explanatory (it names PoPE's mechanism, and the unitarity condition explains our
measured entanglement) but its single quantitative prediction is not supported
here. Tables: DIMSWEEP_d{128,256,512}.md

## Long-T eval (up to 32x training horizon): PoPE is flat, MapWM-Hier COLLAPSES

Existing checkpoints evaluated at T_explore up to 2048 (trained at 64), inference
only, n=3, 100 trials. Motivated by the observation that hier-goal cannot
discriminate above ~0.95, which confounds "hierarchy adds nothing to PoPE" with
"no headroom exists".

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| PoPE-Flat | 0.950 | 0.948 | 0.952 | 0.950 |
| MapPoPE-Flat | 0.950 | 0.949 | 0.951 | 0.946 |
| MapPoPE-Hier | 0.951 | 0.947 | 0.951 | 0.949 |
| MapPoPE-Hier-CoarseIdx | 0.951 | 0.948 | 0.952 | 0.947 |
| MapWM-Hier-CoarsePI | 0.899 | 0.833 | 0.774 | 0.735 |
| MapWM-Hier | 0.856 | 0.704 | 0.567 | **0.542** |
| MapWM-Flat | 0.726 | 0.684 | 0.659 | 0.600 |
| Plain-Flat | 0.601 | 0.578 | 0.539 | 0.553 |

1. **All PoPE variants are flat to 32x training length** (0.946-0.952, ±0.001-0.005).
   They are mutually indistinguishable at EVERY length, so "hierarchy/path
   integration adds nothing to PoPE" is a statement about the task's lack of
   headroom above ~0.95, NOT a measurement of zero contribution. The earlier
   phrasing ("contributes literally nothing") overclaimed.
2. **But the task discriminates BETTER at long T.** PoPE minus best-non-PoPE grows
   from +0.054 (T=256) to +0.213 (T=2048). The ~0.95 plateau is PoPE's ceiling,
   not the task's -- other variants have room to fall, and do.
3. **NEW: the hierarchy advantage REVERSES at extreme length.** At T=256
   MapWM-Hier (0.856) ~= CoarsePI (0.899); by T=2048 MapWM-Hier collapses to
   0.542 while CoarsePI holds 0.735. CoarsePI is the genuinely length-robust
   non-PoPE design; the pooled-angle MapWM-Hier is the LEAST robust MapFormer
   hierarchy variant far out. This corroborates the coarse-angle probe (pooling
   averages away ~40% of the path signal) and means the headline "MapWM-Hier is
   best" holds only within ~4x of training length.
