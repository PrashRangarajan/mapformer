---
name: project-loop-and-correction
description: Looping substitutes for depth and beats the Kalman correction under noise; refining theta does nothing. What Level 1.5 is actually made of.
metadata:
  type: project
---

**REFINING THETA IS DEAD**, tested in the regime built for it (action noise, torus,
4 arms x 3 noise x 3 seeds). refine minus fixed-theta: -0.001/-0.011/+0.005 at
T=128, +0.006/+0.003/-0.005 at T=512. Every |t|<2, NO slope in noise (a positive
slope was pre-registered). Learned gate mean|g| **0.083 with inconsistent sign** —
the optimiser DECLINED to refine, and the gate was verified escapable first. That
plus the sequence-axis finding makes "the Kalman win is stabilisation and
token-type gating, NOT inference" a two-axis result.

**THE CONTROL ARM WON.** At training length vs Vanilla: loop +0.138 (t=12.1) at
p=0.10 and **+0.205 (t=8.8)** at p=0.25; Level15 gives +0.023 and +0.004 on the
same rows. A shared block applied 4x is ~9x more effective under action noise than
the purpose-built correction, at FEWER parameters. n=3, found by reading the
control column — needs pre-registered replication.

Level15's only detectable effect is at OOD LENGTH (+0.025, t=3.82) — stabilisation,
not inference.

**THE LOOP'S COST IS LENGTH, and it is mostly trainable away.** Degradation
128→512: Looped -0.243 vs Vanilla -0.065. But loop count is a RUNTIME knob: same
weights, T=512 peaks at 2 passes and falls to 6. Sampling the count in training
({2..6}) **flattens the count-vs-accuracy curve from 0.178 spread to 0.001** — 0.998
at ONE pass vs 0.821 for the fixed model. 4x cheaper inference for free, and it was
NOT the pre-registered question. The OOD gain (+0.092) is t=1.67 at n=5, i.e. not
established, and does not transfer to noise.

**WHAT LEVEL 1.5 IS MADE OF** (single seed; n=5 replication was running as of
2026-09-01, marker `.l15_ablation_done`, results to `L15_ABLATION.md`):
Level15 1.000/0.993 · DARE 1.000/0.992 · **NoMeas 0.904/0.831 (the wrap alone)** ·
NoCorr 0.940/0.833 (≈vanilla) · **ConstR 0.795/0.672 (no gate — WORSE than nothing)**.
So it does NOT reduce to clamping theta, the per-token gate is load-bearing, and
DARE≈Level15 means the principled Kalman gain is irrelevant. Clean config only —
the lm200 column is under the July retraction.

See [[feedback_premise_before_test]], [[project_hierarchy_negative]].

**Level 1.5's decomposition RESOLVED at n=5 (2026-09-01, L15_ABLATION.md): there
isn't one.** The named parts each come out at no measurable cost when removed
alone -- measurement head (Level15-NoMeas +0.036 loss-matched, t 1.23), per-token
gate (Level15-ConstR +0.000), learned Pi vs DARE (+0.008). RETRACT "removing the
gate is worse than nothing": at n=1 ConstR 0.672 < NoCorr 0.833; at n=5 ConstR
beats NoCorr on 5/5 seeds, sign inverted. WITHDRAW "it does not reduce to clamping
theta" -- unmeasured in both directions.

Rule 9 flipped two readings here, in opposite directions at once. r(loss, acc) =
-0.93/-0.90/-0.81 over 30 runs. RAW, the only detectable contrast was ConstR >
NoCorr -- against the worst-converging arm in the set. LOSS-MATCHED it vanishes
(t 0.51) and Level15 - Vanilla appears instead (+0.062 t 3.08 at T=512, +0.124
t 3.83 at T=1024), where raw it was inside its MDE. The filter's effect is real
and OOD-length-only; the story about which piece does it is not supported.

**Filter x loop are NOT complementary (2026-09-01, n=12, L15_LOOP_2X2.md).** The
2x2 was finally built (Level15Looped, param-matched interaction). Interaction
UNMEASURED at every length, and the levels settle it anyway: the combination is
WORSE than the filter alone at OOD (0.830 vs 0.878 at T=1024). Best OOD arm is
Level15 by itself. The anti-correlated-profiles argument for complementarity was
suggestive and wrong; the pre-registered mechanism objection was right.

Two things worth carrying: (a) **the loop's training-length win is convergence,
not representation** -- raw +0.052 (t 3.03, 12/12) but loss-matched +0.006, with
r(loss,acc) = -0.956 at T=128, and Vanilla's mean final loss 0.1549 vs Looped's
0.0076; (b) **r(loss, accuracy) is LENGTH-DEPENDENT** -- -0.956/-0.471/-0.326 here
vs -0.930/-0.897/-0.812 in the L15 ablation, because the loop arms converge well
and still vary hugely OOD. So the loop's OOD failure is not a convergence failure,
and a loss-matched residual is well justified at training length but weak at OOD.
Check r per length before leaning on it.

**The filter does not pay even where its premise holds (2026-09-02,
MQ_NOISE_2X2.md).** Built stochastic explore transitions into Match-Query so the
recorded action stream drifts from true position (0 -> 13 cells at p=0.10, gated
before training, every shortcut gate held). Level15 - Vanilla is +0.035 at p=0 and
+0.038 at p=0.10 -- **FLAT in drift**, change +0.003. A correction whose benefit
does not scale with the amount to be corrected is not doing correction.
CAVEAT: take 1's arms were far from converged at p=0.10 (final match loss 2.37-2.53
against chance 2.77 and 0.46 for the best arm at p=0), which is a rule-10
violation; take 2 reruns it at lr 1e-3 / 600 ep.

**The LOOP is the real mechanism on this task**: Looped - Vanilla +0.373 (t 4.17)
at p=0 and +0.121 (t 5.22, 8/8) at p=0.10 -- detectable at BOTH noise levels, the
only detectable contrast in the study. And the filter INSIDE the loop is NEGATIVE
(-0.141 at p=0, -0.025 at p=0.10), interaction -0.176 / -0.063. Same direction as
the clean-torus 2x2: they do not compose, and now that holds on a task with a
premise as well as one without.

Clean replication worth noting: Looped at p=0 scored 0.878 +/- 0.092 against
LOOP_HEADROOM's 0.870 +/- 0.099 in a different batch.

