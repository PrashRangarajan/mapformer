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
