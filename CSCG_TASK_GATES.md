# CSCG-derived tasks -- pre-flight gates (CPU, no training)

## Transitive inference (stitch), split by the negative control

Two 8x6 rooms, 15 observation types, shared 3x3 corner patch,
plus a CONFOUNDING identical patch elsewhere in room A.
**chance = 0.0667.**

| start patch | scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |
|---|---|---|---|---|---|---|
| shared | 8.3 | 0.0724 | 0.0702 | 0.0637 | 0.0633 | 0.0750 |
| confound | 8.0 | 0.0752 | 0.0692 | 0.0729 | 0.0700 | 0.0766 |

## Schema transfer (shortcut across unvisited interior)

8x8 room, 20 observation types, periphery-only
exploration. **chance = 0.0500.**

| scored/ep | marginal | last-obs | n-gram o1 | o3 | o5 |
|---|---|---|---|---|---|
| 7.0 | 0.0552 | 0.0514 | 0.0494 | 0.0564 | 0.0570 |

## Verdict: both PASS, no shortcut found

Every baseline sits at the measured chance rate on the first attempt -- the only
task pair today that needed no fix. The two stitch conditions are also BALANCED
(8.3 vs 8.0 scored events per episode), so the negative control is not
systematically easier than the real case and the split metric is meaningful.

## What each task ports, and what it deliberately does not

**Transitive inference (stitch).** Ports CSCG's setup directly: two 8x6 rooms,
15 observation types, a shared 3x3 corner patch, and -- the reason this task is
worth running -- the paper's own negative control:

    "there is another patch in the first room that is identical to the merged
     patches, but was not merged... not simply looking for locally identical
     patches to merge."

An episode starts phase T inside one of the two identical patches, chosen at
random. The local evidence is the same; the correct continuation is not. A model
that merges on appearance alone fails the confound case, and the metric is
reported SPLIT so that failure cannot be averaged away.

**Schema transfer.** Does NOT port CSCG's test, and this is deliberate. Their
test is planning -- "we queried to find the shortest path... the CSCG returned the
correct sequence of actions". Two reasons not to copy it:

  1. MapFormer has no planner; there is no transition graph to run Dijkstra on.
  2. Scoring planner action sequences is the exact failure that voided five tasks
     here on 2026-08-09 (n-grams on the action stream alone scored 0.650-0.969
     against a chance of 0.250 -- `PLANNER_TASK_AUDIT.md`).

Instead it ports the capability the paper states underneath the demo: "shortcut
travels between visited locations through locations that have never been visited".
The agent sees ONLY the periphery, then crosses the unobserved interior blind and
must recognise where it re-emerges. Interior cells are never scored -- their
observations are genuinely unknowable, so the ceiling stays at 1.0 and the metric
cannot be inflated by guessing there.

Deviation from CSCG in both: CSCG runs EM over 10,000-step sequences per room.
MapFormer builds maps in context, so each episode contains the whole experience
and the layouts are redrawn every episode.
