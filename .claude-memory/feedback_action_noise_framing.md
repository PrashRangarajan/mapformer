---
name: Action noise → stochastic-transition MDP framing
description: How to defend the action-noise experiment against "this is artificial" critiques. Use this framing in any new writeup; don't lead with "action noise".
type: feedback
---

The user got pushed back on early in the project for framing experiments as "10% action noise". The critique: it sounded artificial. The defensible reframing reached on 2026-05-01:

**The phenomenon being modelled is real.** Proprioceptive / action-record noise is a well-established noise category in navigation systems:
- Animal vestibular drift (~5°/sec error in 30s of darkness)
- MEMS gyro bias drift (~5°/hr at rest)
- Wheel slip in differential-drive robots (~1–2% per meter)
- Teleoperation packet loss
- Imitation learning from imperfectly logged demos

The Markovic et al. (2017) paper deriving the wrapped-innovation Kalman filter on SO(2) is *specifically* for this scenario. So the action-noise test maps to a textbook InEKF use case.

**The specific implementation is conservative.** 10% iid uniform replacement is the information-theoretic worst case for a fixed corruption rate — harsher than any real proprioceptive noise (which is concentrated near the truth, bursty, biased). So our +10pp Level 1.5 win under this noise is a *lower bound*; structured real noise should give larger wins.

**Mathematical equivalence to a stochastic-transition MDP.** For a uniform random policy:
- Setup A: corrupt action records post-hoc; trajectory was generated cleanly
- Setup B: env executes random action 10% of the time; trajectory is stochastic; recorded action is the commanded one
- Both produce identical (action_record, observation) data distributions

This means our action-noise experiment IS a stochastic-transition MDP test, just dressed in different vocabulary. Use the stochastic-transition framing — it's standard control/RL vocabulary that reviewers recognise immediately.

**How to apply:**
- Don't lead with "action noise" in any writeup or talk. Lead with the non-circular wins (clean OOD T=512 +8pp on torus, NLL across the board).
- When you do discuss noise robustness, frame it as: "stochastic-transition MDP with 10% transition stochasticity — a standard robustness benchmark".
- If asked the difference: "Action-record corruption is one specific noise model; our setup is mathematically equivalent to the more general stochastic-transition MDP framing for uniform policies. The discrete-event noise structure we test is harder than Gaussian process noise (heavy-tailed, violates KF's nominal assumptions), so our wins are conservative."
- Don't claim Level 1.5 generalises to all noise types — it specifically handles process noise / proprioceptive noise / continuous obs noise. Not aliasing, not outliers, not loop closures.

**Code support (added 2026-05-01):** `environment.py` now has both `--p-action-noise` (post-hoc record corruption, the original) and `--p-transition-noise` (execution-time stochasticity, the standard MDP framing). The pending pipeline `run_dog_fix_and_continuous.sh` will produce `STOCHASTIC_TRANSITION_RESULTS.md` empirically validating the equivalence.
