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
