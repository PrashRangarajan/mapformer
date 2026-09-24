---
name: feedback-validate-task-first
description: Gate the task, audit the design and check the premise before any GPU (CLAUDE.md rules 11-13). Unchecked setup assumptions, not model bugs, wasted most batches; the user asks for audits.
metadata:
  type: feedback
---

The why behind CLAUDE.md rules 11-13.

## Validate the TASK and the COMPARISON (2026-07-16)

Three invalid setups in one session, same root cause: the MODEL was validated carefully (causality,
param-matching, seeds) while the TASK and the COMPARISON were not.
- **Cascade "win"**: the baseline was a stale, non-converged April checkpoint
  ([[feedback-convergence-first]]).
- **Aggregate "win"**: flat attention had merely trained at a shorter sequence length. Retracted.
- **Rooms open-plan "planning"**: 100% of BFS-optimal actions were greedy, so there was no planning
  problem; the flat-vs-hierarchical tie was vacuous.

**Why:** each was detectable in minutes of CPU and was checked only after hours of GPU.

**How to apply -- `validate_task.py` before training anything new:**
1. `trivial_baseline`: can greedy / majority / recency solve it? Then it does not test the capability.
2. `label_stats`: chance, label entropy, majority-class frequency.
3. `demand_profile`: where the evidence lives. At T=256, 95% of revisits are within 64 steps, which
   is why the bounded-memory prediction failed, knowably in advance.
4. Confounds run IN THE SAME BATCH as the main arms: params, training length, stale baselines,
   RNG/init drift, capacity, component attribution.
Validated discrimination: rooms_open FAIL (greedy 1.000), rooms_maze_full WARN (0.949), rooms_maze
tree PASS (0.704, 2.87x detours). **Beware motivated task design**: by the third environment built to
let hierarchy win, a win is p-hacking with environments. Pre-commit to the fair test and accept its
negative ([[project-hierarchy-negative]]).

## Audit the design before launch (2026-09-23)

Before a batch, have an independent audit check: does the changed flag change only what it claims
(`--n-steps` also changes task composition), are tokens/steps/params matched, are scored-position
counts and floors known, does the eval reproduce a stored number exactly, do the arms' losses overlap.
**Why:** the Dyck width confound (1L d=64 vs 2L d=128), the frequency-ladder confound (index base
10000 vs path arms from grid_size) and the occupancy-blind GPU picker were caught only after GPU time.
The rank matched-length audit found before any GPU that old r=2/r=4 losses never overlapped, 94% of
the headline sat in one stratum (short-gap revisits late in the sequence) and wrap revisits were below
floor. The user asks for this step and for results verified before they are relayed.
**How to apply:** write the pre-registration, run the audit (an agent is fine), record its verdict in
the prereg, then launch. An eval-only stratification of existing checkpoints is usually free.

## Check the premise; prefer runtime knobs; split hypotheses (2026-08-31)

1. **Premise.** A theta-refinement loop was tested on Match-Query, where actions are CLEAN (no drift)
   and the query phase is BLIND (nothing to correct with); the repo already said so. 16 runs to
   replicate a known negative. Name the condition a mechanism needs and verify the task supplies it.
2. **Runtime knob.** I specified 12 training runs to test loop count; it is a forward-pass argument.
   The eval-only sweep took 90 s and found the mechanism (T=512 peaks at 2 passes, T=128 at 4).
3. **Split.** "Iteration compounds the damage VIA residual growth" was tested as one claim; the
   residual half failed, I retracted all of it, and the iteration half was right.
`train_hourglass_enwik8.py` once saved no checkpoints; it now has opt-in `--save-ckpt`/`--data-val`.

Related: [[feedback-scheduler-and-measurement-traps]], [[project-robustness-vs-capability]].
