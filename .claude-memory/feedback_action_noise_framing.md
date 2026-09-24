---
name: feedback-action-noise-framing
description: Vocabulary for noisy-action experiments -- call it a stochastic-transition MDP, not "10% action noise". Vocabulary only; it makes no claim that Level 1.5 wins under noise.
metadata:
  type: feedback
---

Early in the project the user was pushed back on for framing experiments as "10% action noise": it
sounded artificial. The defensible framing (2026-05-01):

**The phenomenon is real.** Proprioceptive / action-record noise is a standard category in
navigation: vestibular drift (~5 deg/s after 30 s of darkness), MEMS gyro bias (~5 deg/hr at rest),
wheel slip (~1-2% per metre), teleoperation packet loss, imperfectly logged demonstrations. Markovic
et al. (2017), the wrapped-innovation Kalman filter on SO(2), is written for exactly this case.

**It is a stochastic-transition MDP.** For a uniform random policy, (A) corrupting action records
after a clean rollout and (B) an environment that executes a random action with probability p while
recording the commanded one produce identical (action_record, observation) distributions. Use the
stochastic-transition vocabulary; reviewers recognise it. `environment.py` has both knobs:
`--p-action-noise` (post-hoc record corruption) and `--p-transition-noise` (execution-time). The
promised empirical equivalence file (`STOCHASTIC_TRANSITION_RESULTS.md`) never landed; the argument is
analytic. Match-Query also has stochastic explore transitions (`MQ_NOISE_2X2*.md`).

**How to apply:**
- Do not lead a writeup with "action noise"; say "stochastic-transition MDP with p transition
  stochasticity". Uniform replacement is a discrete, heavy-tailed noise model, harsher than Gaussian
  process noise; say that when asked why this model.
- **Make no performance claim from the framing.** The old "our +10pp Level 1.5 win is a lower bound"
  is deleted: Level 1.5's benefit does not grow with drift (flat, then -0.141, across two recipes,
  `MQ_NOISE_2X2*.md`), its torus effect is OOD-length only, and the loop beats it under noise
  ([[project-loop-and-correction]]).
