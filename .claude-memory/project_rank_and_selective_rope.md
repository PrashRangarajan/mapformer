---
name: project-rank-and-selective-rope
description: r=2 is under-provisioned (r=4 buys +0.085 for 384 params); Selective RoPE occupies the same slot and is no better; the paper's Fig 4 reproduces 3 of 4.
metadata:
  type: project
---

**r=4 IS THE DEFAULT TO USE (2026-09-04, RANK_SWEEP.md).** Torus, 8 seeds, one
batch: against r=2 at T=1024, r4 **+0.085** (t 3.57, 8/8, sign p=0.008), r8 +0.091,
r16 +0.079, r32 +0.095. A **STEP at r=2, not a slope** -- flat from r=4 up. Costs
+384 params (+0.19%) and also cuts seed sd 0.064 -> 0.012 at T=1024, which matters
because sd sets every MDE in this project.

**WHY: the r=2 code is SKEWED, not too small** (ACTION_GEOMETRY.md). Given 4 dims
the model puts its actions in a 2-plane anyway (100.0% of energy; 99.96% at r=32),
so the paper's dimensional argument is right about what is EXPRESSIBLE. But at r=2
opposite actions fail to cancel by 0.495 of the action scale and |cos(N,E)| = 0.783
-- north and east nearly PARALLEL. At r=4: 0.092 and 0.175. Interpretability
survives: project onto the top two singular directions.

**PAPER FIG. 4 REPRODUCTION (PAPER_FIG4_REPRO.md), 8 seeds at r=2:**
- C1 ||D_act||/||D_obs|| = 25.0 -- reproduces
- C2 cos(opposite) = -0.729 +/- 0.373 vs paper's -1 -- weak; r=4 gives -0.996
- C3 |cos(orthogonal)| = 0.779 -- **reproduces the paper's OWN reported limitation**
  (their caption proposes bounded-energy constraints as the fix). **r=4 does that
  job for free: 0.779 -> 0.174, no regulariser.**
- C4 ||v_obs||/||v_act|| = **0.57, INVERTED** vs the paper's >>1. Hypothesis being
  tested: Fig. 4 is an EM model (Sec 5.4 is explicitly about EM's separate pools),
  and MapWM's additive attention has no such split. See run_em_fig4.sh.

**SELECTIVE ROPE IS THE SAME SLOT AND NO BETTER HERE (SELECTIVE_ROPE.md).** Its
`temp*cumsum(conv1d(W_omega q))` and MapFormer's `omega*cumsum(W_out W_in x)` both
drive the PHASE; PoPE is the orthogonal half (magnitude). Full generator: parity
-0.009, torus +0.031/+0.048, at +8.2% params. Its three knobs FLIP SIGN between
tasks (conv -0.020/-0.064; no-bottleneck -0.020/+0.058; gate -0.030/+0.086). The
two ~8k knobs are indistinguishable from each other, and
**gate-as-token-suppressor is FALSIFIED** (GATE_PROBE.md: 1.35x on the torus where
it helps, 1.54x on parity where it hurts).

Priority: SRoPE 21 Nov 2025, MapFormer 24 Nov, neither cites the other.
See [[reference-language-and-pope]] for the polar decomposition these sit in.
