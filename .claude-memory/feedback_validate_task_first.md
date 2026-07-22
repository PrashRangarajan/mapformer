---
name: feedback_validate_task_first
description: "Validate the TASK and the COMPARISON before spending GPU — three invalid setups in one session all came from unchecked setup assumptions, not model bugs. Run validate_task.py first."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d17148dd-e54f-48b7-8a88-096f71aadfc5
  modified: 2026-07-22T05:48:46.516Z
---

Three invalid experimental setups in a single session (2026-07-16), all with
the same root cause: the MODEL was validated carefully (causality checks,
param-matching, multi-seed) while the TASK and the COMPARISON were not
validated at all.

- **Cascade "win"** — the baseline was a stale, non-converged April checkpoint.
  (see [[feedback_lm200_stuck_baselines]])
- **Aggregate "win"** — flat attention wasn't incapable of aggregation; it had
  merely trained at a shorter sequence length. Retracted.
- **Rooms open-plan "planning" task** — 100% of BFS-optimal actions were
  greedy (distance-reducing), so there was no planning problem at all. The
  flat-vs-hierarchical tie was vacuous.

**Why:** every failure was an assumption about the setup, and each was
detectable in minutes of CPU — but only got checked after hours of GPU.

**How to apply — run `validate_task.py` BEFORE training anything new:**
1. `trivial_baseline` — can greedy / majority-class / recency already solve it?
   If yes, the task does not test the capability you think it does.
2. `label_stats` — chance level, label entropy, majority-class frequency.
3. `demand_profile` — WHERE the evidence lives (e.g. revisit-lag distribution).
   At T=256, 95% of revisits are within 64 steps, so long-range memory is
   barely exercised — this is exactly why the bounded-memory prediction failed,
   and it was knowable in advance.
4. Confound checklist, all run IN THE SAME BATCH as the main arms (never only
   after a positive result): parameter count, training length, stale baselines,
   RNG/init drift, capacity budget, component attribution.

Validated discrimination: rooms_open FAIL (greedy 1.000), rooms_maze_full WARN
(0.949), rooms_maze tree PASS (0.704, 2.87x detours).

**Also beware motivated task design.** By the third environment built
specifically to let hierarchy win, a win would be p-hacking with environments.
Pre-commit to which task counts as the fair test, and accept the negative if it
loses there — the mechanism-backed negative is the stronger contribution.
See [[project_hierarchy_negative]].
