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

## Honest priors

Every prior hierarchy attempt here was a clean negative, but all were on
exact-recall or used position-based pooling. This task is the first with a
genuine motif-level sufficient statistic AND a demonstrated long-range demand.
If `Hourglass_MotifSeg` still doesn't beat flat, that is a strong, mechanism-
level negative (hierarchy fails even where its precondition is met). If it wins
while `Hourglass_k2` does not, that isolates collapse-by-structure as the
operative mechanism — the cleanest positive result this line could produce.
```
