# How far back must each task reach?

`REVISIT_DISTANCE.md` measured that index-position models beat the blank floor ONLY at recurrence interval 1-2 (+0.05 to +0.07), and sit at or below it in every other bucket. **Attention path-integrates over a horizon of roughly two steps.**

If that is the whole story, an index model's performance on a task should be predicted by the share of that task's scored events whose answer lies inside the horizon. That share is computed here from each task's own generator, with no model involved.

| task | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ | median | **within horizon (1-2)** |
|---|---|---|---|---|---|---|---|---|---|
| paper task (T=128) | 0.191 | 0.165 | 0.248 | 0.254 | 0.092 | 0.034 | 0.016 | 8 | **0.191** |
| paper task (T=512) | 0.156 | 0.135 | 0.208 | 0.222 | 0.090 | 0.043 | 0.146 | 10 | **0.156** |
| Match-Query (TE=512 TQ=256) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 517 | **0.000** |
| family tree (depth 5, T=64) | 0.362 | 0.180 | 0.167 | 0.137 | 0.099 | 0.055 | 0.000 | 4 | **0.362** |

## Result: the single-variable horizon account FAILS

The prediction was that an index model's share of recoverable headroom would
track the share of scored events lying inside the ~2-step horizon. Put the two
side by side and it does not:

| task | within horizon | index model recovers | consistent? |
|---|---|---|---|
| family tree | **0.362** | **79%** of model-achievable headroom (0.612 vs floor 0.163, ceiling 0.728) | over-recovers by 2x |
| paper task | 0.191 | **1.7%** (0.514 vs floor 0.506, ceiling 0.989) | under-recovers 10x |
| Match-Query | **0.000** | **8.1%** (0.154 vs floor 0.089, ceiling 0.888) | recovers with ZERO in-horizon events |

The ordering is not even monotone: Match-Query has no in-horizon events at all
and still recovers more headroom than the paper task, which has 19% of them.
**So "fraction of events within the horizon" does not predict cross-task
performance, and the tidy version of this account is wrong.**

Why it was worth measuring anyway: this cost ten minutes of CPU and it kills a
hypothesis that would otherwise have gone into a paper as its central claim.

## What survives

The horizon itself is a real, replicated MEASUREMENT (`REVISIT_DISTANCE.md`,
n=3, two architectures): index models beat the blank floor at recurrence
interval 1-2 and nowhere else, on the paper task. That is a within-task fact
about where their competence lives, and it is not in question here.

What fails is the extrapolation -- treating that one number as sufficient to
predict behaviour on other tasks. At least three things differ across these
tasks and are not controlled: the floor's construction (blank majority vs a
visit-frequency hub baseline), the degree of observation aliasing, and whether
the answer is reachable by non-positional structure at all. The family tree's
0.163 hub floor in particular probably understates what a non-positional
strategy can do on a 63-node graph with 8 observation types, which would inflate
its apparent recovery.

## What this redirects the horizon measurement toward

Measure `h` WITHIN one task, across architecture and scale, where floors and
aliasing are held fixed:

  - does `h` grow with depth, width, or training budget?
  - if it saturates, the horizon is architectural -- the strong result
  - if it grows with scale, it is a capacity story -- weaker, still publishable,
    and honest either way

That question is well posed and unaffected by what failed here. The cross-task
law is not, and should not be claimed without a controlled way to equate floors
across tasks.

## Why compositional is excluded, and why that is the point

`cross_nb` scores prediction in a room-instance the agent has NOT visited, using
a motif seen in a different instance. That is template matching, not reaching
back to a previous visit, so recurrence interval is undefined for it. That
remains the cleanest available explanation for why index models reach ~80% of
MapFormer's score there and sit on the floor on the paper task: **some tasks
never ask attention to path-integrate at all.** It is a mechanism claim about one
contrast, not a general law.
