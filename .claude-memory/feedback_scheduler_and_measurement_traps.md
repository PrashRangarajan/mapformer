---
name: feedback-scheduler-and-measurement-traps
description: Four traps bought in one week — fill-first GPU pickers, inferring accuracy from loss, wide pre-registered bands, editing running scripts.
metadata:
  type: feedback
---

Four ways I wasted real time on the MapFormer runs, each caught only by looking at
something other than the log.

**1. A fill-first GPU picker is a single-GPU scheduler when jobs <= MAXPG.**
`if on_gpu(0) < MAXPG -> 0 elif on_gpu(1) < MAXPG -> 1` was fine at MAXPG=3. After
`--fast-attn` cut memory to ~2.1 GiB/job I raised it to 6, and with exactly 6 jobs
GPU 0 never filled: **one 4090 sat at 0% for three hours** while the log happily
reported six successful launches. **Why:** the optimisation created the bottleneck.
**How to apply:** pick the LESS LOADED device. And do NOT then interleave job types
to "balance" — alternating types against an alternating picker phase-locks and puts
every long job on one device. I did exactly that and reproduced the bug I was fixing.
Check `nvidia-smi`, never the launch log.

**2. Do not infer held-out accuracy from training loss.** I predicted a condition
would be null because its index arm reached 0.03 training loss; it scored **0.674
held-out against 0.949**. **Why:** the r=-0.996 affine relation between loss and
accuracy holds in some regimes and not others, and both arms being at low loss says
nothing about the gap. **How to apply:** wait for the eval. Never trail a prediction
off a loss curve.

**3. A wide pre-registered band is not a pre-registration.** My three outcomes were
"between -0.010 and +0.374", "near -0.010", "at/above +0.374". The middle case
mechanically fired on +0.015, which is 0.025 from the null reference against a 0.150
noise floor — i.e. identical to it. **How to apply:** set branch boundaries against
the MEASURED NOISE FLOOR, not against the endpoints, and make the branches disjoint.

**4. Editing a running bash script is unsafe.** Bash reads by byte offset; an insert
before the read point makes it resume mid-token. Kill and relaunch — cheap when the
script is parked in a wait loop, and children survive killing the parent.

Related: [[feedback_convergence_first]], [[feedback_cwd_aggregator_bug]].
