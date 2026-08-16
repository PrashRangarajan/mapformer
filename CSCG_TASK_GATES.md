> **NOT READY TO RUN (2026-08-09).** An independent audit found four
> result-invalidating problems in these two tasks. Two are fixed, two are not.
> Do not train against them until the remaining two are resolved.
>
> **FIXED**
> - Stitch confounder sat at a grid corner while the join sat interior, giving
>   valid-action counts [2,3,3,3,4,4,3,4,4] vs [4,4,4,4,4,4,4,4,4]. Moved to A's
>   interior; both now [4]*9.
> - The gate suite had no condition-identifiability test at all. Added.
>
> **NOT FIXED -- these void the tasks as designed**
> 1. **The negative control is still defeatable without a map: 0.617 balanced
>    accuracy** (was 0.762 before the fix, chance 0.500). The cue is relative
>    displacement -- *the mechanism under test* -- so the task hands the
>    path-integration arm a free win on the discrimination the control exists to
>    make. This is structural: from the join 87 cells are reachable, from anywhere
>    inside room A only 48, and that asymmetry leaks into wall-collision
>    statistics wherever the confounder is placed.
> 2. **Only ~14% of "shared" scored events actually require stitching**, and ~70%
>    of shared episodes contain none at all. The shared patch is covered by BOTH
>    phase walks, so predicting cells near it needs one room's map, not a merge.
>    The headline would be ~86% non-transitive.
> 3. **`environment_schema.py` is not schema transfer.** CSCG's schema is Room 1's
>    *transition matrix reused in Room 2*. This env has ONE room and no structural
>    reuse. What it measures is blind path integration on a bounded grid -- which
>    `Match-Query` already measures, better and with more validation.
> 4. **Its "shortcut across the unvisited interior" is 71.9% plain border steps**
>    (measured): the walk is unconstrained, so nothing forces an interior crossing
>    before scoring.

# CSCG-derived tasks -- pre-flight gates (CPU, no training)

## Transitive inference (stitch), split by the negative control

Two 8x6 rooms, 15 observation types, shared 3x3 corner patch,
plus a CONFOUNDING identical patch elsewhere in room A.
**chance = 0.0667.**

| start patch | scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |
|---|---|---|---|---|---|---|
| shared | 9.7 | 0.0710 | 0.0607 | 0.0650 | 0.0663 | 0.0709 |
| confound | 8.5 | 0.0757 | 0.0689 | 0.0607 | 0.0631 | 0.0627 |

### Condition-identifiability from the ACTION STREAM ALONE

displacement cue fires: shared **0.537**, confound **0.302** -> balanced accuracy **0.617** (chance 0.500).
Above ~0.55 means the negative control is defeatable without a map, by relative displacement -- the mechanism under test. VOID if so.


## Schema transfer (shortcut across unvisited interior)

8x8 room, 20 observation types, periphery-only
exploration. **chance = 0.0500.**

| scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |
|---|---|---|---|---|---|
| 6.9 | 0.0559 | 0.0556 | 0.0417 | 0.0534 | 0.0581 |
