# Compositional-motif experiment: does a motif-collapsing hierarchy earn its keep?

## The question this answers

From the design discussion: in a compositional environment with repeatable
motifs, should two instances of the same motif at different locations map to
the **same** point at the higher level (structural abstraction, enables reuse)
or **different** points (absolute localisation, enables exact recall)? The
answer is task-dependent — hierarchy helps iff a lossy summary is a *sufficient
statistic*. This experiment builds an environment where a clean
**motif-level** target exists and tests whether a hierarchy that COLLAPSES
motif instances beats flat MapFormer, while a hierarchy that keeps them
DISTINCT (absolute position) does not.

This is the first regime tested here that is *predicted* to favour hierarchy —
in contrast to the exact-recall torus task, which provably resists it.

## Environment (`environment_compositional.py`)

64×64 torus tiled into 8×8 rooms (64 rooms). `n_templates=4` motif templates,
each an 8×8 obs pattern; each template assigned to several rooms at DIFFERENT
locations. Obs at a cell is fully determined by (template, local-position), so
two rooms sharing a template emit the same pattern. **Templates + assignment
are redrawn every episode** (`fresh_per_episode=True`) — no cross-episode
memorisation; motif structure must be inferred in context.

Three obs-position masks:
- `revisit_mask` — exact absolute cell seen before (fine target; paper-standard).
- `motif_revisit_mask` — motif-cell (template, local) seen before, any copy.
- `cross_instance_mask` — motif-cell seen before but exact cell NOT (**the
  compositional target**: predictable only by recognising two rooms share a
  template and reusing the pattern).

## Task validity (VALIDATED before training — `validate_compositional.py`)

| Property | T=128 | T=256 | T=512 |
|---|---|---|---|
| cross-instance label mass | 14.3% | 23.2% | 34.4% |
| env consistency failures | 0 | 0 | 0 |
| copy-nearest-motif ORACLE acc | 100% | 100% | 100% |
| majority (blank) baseline | ~50% | ~50% | ~50% |
| cross-instance lag median (steps) | 26 | 72 | 160 |
| cross-instance solvable in ≤32-step window | 57% | 30% | 16% |

Reads as: floor ≈ 50% (blank-dominated), ceiling = 100%, target deterministic,
disjoint from exact-revisit, and the matching evidence sits far back and gets
farther with length — so a bounded local window cannot solve it. This is NOT
the LocalOnly trap; long-range motif memory is genuinely required.

Metric refinement: because blank is ~50% of cells, primary metric is accuracy
AND NLL on **non-blank cross-instance** positions (the discriminating subset);
overall cross-instance accuracy reported alongside.

## Hypotheses (falsifiable, tied to the mechanism)

- **H1 (flat baseline may already transfer):** flat MapFormer-WM's content
  attention (A_X) can match on obs content across rooms, giving some
  cross-instance transfer for free. If flat already ceilings, hierarchy has no
  room to help — an honest null. WM (additive OR) is expected to transfer more
  than EM (multiplicative AND-gate suppresses cross-instance via absolute-θ
  mismatch): predicts **WM > EM on cross-instance, EM ≥ WM on exact**.
- **H2 (absolute-θ Hourglass does NOT exploit compositionality):** the built
  `Hourglass_k2` pools on a FIXED token stride and carries the ABSOLUTE
  path-integrated angle, so its coarse tokens are location-specific and do not
  align to rooms. Predicted ≈ flat (or worse — the prior hierarchy negatives).
- **H3 (motif-collapsing Hourglass DOES help):** a variant that (a) segments at
  ROOM BOUNDARIES (one coarse token per room-visit) and (b) represents a room
  by its CONTENT (motif identity), collapsing instances of the same template,
  should beat flat on cross-instance at long T — because the coarse memory is
  then a short sequence of motif codes, exactly the sufficient statistic.
  If H3 holds and H2 does not, we've isolated *why* hierarchy helps:
  **collapse-by-structure, not pool-by-position.**

## Variants to compare

Fine control (exact-recall) run in parallel so the dissociation is legible.

1. `Vanilla` (flat MapFormer-WM, n_layers matched to Hourglass depth) — baseline.
2. `VanillaEM` — tests the AND-gate prediction (H1).
3. `Hourglass_k2` (absolute-θ, fixed-stride) — already built (H2).
4. `Hourglass_MotifSeg` (NEW) — room-boundary segmentation + content-summary
   coarse tokens; collapses motif instances (H3). Two sub-variants:
   - oracle segmentation (uses `step_new_room` — privileged, upper bound)
   - learned/dynamic segmentation (fair; only if oracle wins).
5. Controls the project rule requires:
   - **training-length control:** train flat at the eval length; a Hourglass
     win must survive it (not be mere length-extrapolation).
   - **LocalOnly ablation:** bounded-window flat — must lose on cross-instance
     (validator predicts it will), confirming the win is retrieval not span.
   - **coarse-contribution diagnostic** (`coarse_contribution()` /
     ablate upsample): guard against the inert-coarse failure mode.

## Metrics

- Cross-instance accuracy + NLL (overall and non-blank) — the compositional target.
- Exact-revisit accuracy + NLL — the fine control target.
- Length generalisation: train T=256, eval T ∈ {256, 512, 1024, 2048}.
- Motif-transfer curve: cross-instance accuracy vs lag bucket (does the model
  retrieve far-back motif matches, where the win must come from?).

## Segmentation is the crux (why the built Hourglass is not enough)

The agent wanders room-to-room irregularly, so fixed-stride pooling (current
`Hourglass_k2`) does NOT align coarse tokens to motifs — it pools across room
boundaries and destroys the reuse. The motif-collapsing variant needs
**boundary-aligned (dynamic) segmentation** driven by `step_new_room`, plus a
per-room content readout (one coarse token = one motif code) and, ideally, a
LOCAL coordinate frame reset at room entry so identical motifs produce
identical fine codes. This is TEM's "path-integrate within a room, remap
between rooms" made concrete, and the SpaceTimeHier docstring's flagged-but-
unbuilt "segment by REGION TRANSITION" fix.

## Phased plan (no GPU spent until each gate clears)

- **Gate A (DONE):** task validated — well-posed, long-range, dissociable.
- **Gate B:** enwik8 scaffold sanity-check passes (Hourglass ≈/> Flat-10 bpc at
  equal params, less compute). In flight.
- **Phase 1:** train `Vanilla`, `VanillaEM`, `Hourglass_k2`, LocalOnly on the
  compositional task, single seed, T=256; report the dissociation + whether
  flat already transfers (H1) and whether absolute-θ Hourglass helps (H2).
- **Phase 2 (only if H1 leaves headroom):** build + train `Hourglass_MotifSeg`
  (oracle segmentation first) to test H3. Add training-length control and
  coarse-contribution diagnostic.
- **Phase 3:** multi-seed the surviving contrast; length-gen curves; write up.

## RESULTS — multi-seed, n=3 (updated 2026-07-25, koopman server)

Phase 1 + Phase 3 (multi-seed) run together: 6 variants × seeds {0,1,2},
trained at T=256, evaluated on a **fresh held-out env (seed=10000)** at
T ∈ {256,512,1024,2048}. Added two non-MapFormer controls (`Plain-Hier`,
`Plain-Flat`) to isolate what MapFormer's path-integration bias buys over
ordinary index-RoPE. Raw table: `COMPOSITIONAL_MULTISEED.md`.

Model naming (backbone × structure; old key in parens):
`MapWM-Flat` (Vanilla), `MapEM-Flat` (VanillaEM), `MapWM-Hier` (Hourglass_k2),
`MapWM-FlatHG` (HourglassFlat3), `Plain-Hier` (PlainHourglass),
`Plain-Flat` (PlainFlat).

`cross_nb_acc` (compositional target), mean±std over seeds:

| variant | T=256 | T=512 | T=2048 |
|---|---|---|---|
| MapWM-Hier   | 0.423 ± 0.144 | 0.314 ± 0.144 | 0.166 ± 0.174 |
| Plain-Hier   | 0.324 ± 0.034 | 0.208 ± 0.037 | 0.046 ± 0.012 |
| MapWM-FlatHG | 0.281 ± 0.049 | 0.163 ± 0.040 | 0.037 ± 0.020 |
| MapWM-MotifSeg | 0.254 ± 0.014 | 0.133 ± 0.009 | 0.026 ± 0.006 |
| MapWM-Flat   | 0.270 ± 0.030 | 0.164 ± 0.021 | 0.048 ± 0.012 |
| Plain-Flat   | 0.213 ± 0.001 | 0.100 ± 0.002 | 0.018 ± 0.001 |
| MapEM-Flat   | 0.097 ± 0.013 | 0.047 ± 0.012 | 0.015 ± 0.010 |

**1. Hierarchy helps — in BOTH backbones (robust).** Hourglass beats flat on
all 3 seeds, paired, in the MapFormer family (MapWM-Hier > MapWM-FlatHG:
Δ = +0.09 / +0.28 / +0.06 at T=256) AND the plain family (Plain-Hier >
Plain-Flat: Δ = +0.15 / +0.12 / +0.07). This is the cleanest positive of the
line — and, unexpectedly, it does NOT require MapFormer. **H2 is FALSIFIED:**
the fixed-stride absolute-θ Hourglass was predicted to be ≈ flat or worse; it
beats flat.

**2. MapFormer barely helps over a plain transformer.** Matched pairs: in the
flat setting MapFormer gives a small, consistent edge (MapWM-Flat 0.270 vs
Plain-Flat 0.213, ~+0.05); in the hierarchy setting the edge is seed-dependent
and vanishes on 2/3 seeds (MapWM-Hier vs Plain-Hier paired Δ = −0.01 / +0.29 /
+0.02 — carried entirely by seed1). And `MapEM-Flat` (0.097) is **worse** than
a plain transformer (0.213): the EM AND-gate actively suppresses compositional
transfer. The relayed prediction that plain models sit at the ~0.06 chance
floor is **FALSIFIED** — because the action stream is in the input, a plain
transformer learns to path-integrate via attention. MapFormer's SO(2) code is
an inductive bias, not privileged information, and on this task it is close to
free-riding on the hierarchy.

**3. High variance in MapWM-Hier.** Its mean lead is inflated by one lucky seed
(seed1: 0.625 at T=256 / 0.412 at T=2048 vs ~0.30 / 0.04 for seeds 0,2); std >
gap. The plain family shows the SAME hierarchy effect with 5–30× lower variance,
so the *clean* demonstration of "hierarchy helps" lives in the plain models.

**4. H1 confirmed** (flat already transfers), **WM > EM on cross-instance**
(0.270 vs 0.097). The predicted "EM ≥ WM on exact" did NOT hold here
(MapEM-Flat exact 0.788 ± 0.168 < MapWM-Flat 0.924) — but EM's exact-recall
variance is large (init instability documented in CLAUDE.md), so treat that as
unstable rather than a clean refutation.

**5. enwik8 scaffold (Gate B):** hourglass val_bpc ≈ 2.00 vs flat10 ≈ 2.07 at
equal params, seq=2048, 8000 iters — Hourglass ≥ flat at lower compute,
reproducing the Nawrot et al. efficiency property. Scaffold is sound.

**6. Phase 2 (H3) — oracle motif-segmentation does NOT help (built 2026-07-25).**
`MapWM-MotifSeg`: identical to `MapWM-Hier` (same 600,917 params) except it pools
on ORACLE room boundaries (one coarse token per room-visit) instead of a fixed
stride. Result: **0.254 ± 0.014** at T=256 — *below* the flat control
(`MapWM-FlatHG` 0.281) and far below `MapWM-Hier`; second-worst hierarchy variant
(only EM is worse). It trains fine (`exact_acc` 0.943, healthy) and is causal
(verified) — the failure is specific to compositional transfer. **Strong caveat:**
this v1 tests segmentation ALIGNMENT only; it omits H3 ingredient 3, the
LOCAL-FRAME-RESET. The coarse room-summary is a mean of MapFormer hidden states
that still carry ABSOLUTE path-integrated position, so identical motifs at
different locations do NOT collapse to the same code — the motif-level sufficient
statistic is never formed, and the ~8-token/256-step compression is pure loss with
no abstraction payoff. So this falsifies *"room-aligned pooling helps"* but NOT yet
*"collapse-by-structure helps"* — the decisive test is a v2 with the frame-reset.

**How WM helps vs how hierarchy helps (matched pairs) — the clean summary.** The
two additions help ORTHOGONAL metrics:
- **WM (path integration) → exact positional recall, growing with length.** vs
  plain RoPE at matched structure, `exact_acc` gap: flat +0.02/+0.10/+0.14/+0.11
  and hier +0.03/+0.10/+0.18/+0.17 across T=256..2048 (widens with T = the cumsum
  extrapolation signature). On the content-driven `cross_nb` target WM barely
  helps (+0.03–0.06 flat; seed-noise in hier), and MapEM-Flat is WORSE than plain.
- **Hierarchy → compositional transfer, backbone-independent.** vs flat at matched
  backbone, `cross_nb` gap ~+0.11 (clean/low-variance in the plain family; positive
  but high-variance in MapFormer); it adds ~nothing on `exact_acc`. It does NOT
  require MapFormer, and making the segmentation smart (MotifSeg) did NOT help — so
  the benefit is GENERIC multi-scale compression, not task-structure alignment.

**Caveats / open:** n=3; absolute compositional accuracies are small (0.2–0.4);
`MapWM-Hier` is high-variance (one lucky seed). The decisive remaining H3 test is a
v2 MotifSeg WITH the local-frame-reset (make identical motifs collapse). Reproduce
with `run_comp_multiseed.sh` / `run_motifseg.sh` → `agg_comp_multiseed.py`.

## Honest priors

Every prior hierarchy attempt here was a clean negative, but all were on
exact-recall or used position-based pooling. This task is the first with a
genuine motif-level sufficient statistic AND a demonstrated long-range demand.
If `Hourglass_MotifSeg` still doesn't beat flat, that is a strong, mechanism-
level negative (hierarchy fails even where its precondition is met). If it wins
while `Hourglass_k2` does not, that isolates collapse-by-structure as the
operative mechanism — the cleanest positive result this line could produce.
```
