---
name: project-rank-and-selective-rope
description: It is the PER-HEAD rank of the content-to-angle map: rank 2 per head solves the T=1024 torus on 0-2/8 seeds, rank 3 on 6/8, rank 4 on 8/8; 2x the budget does not rescue rank 2. Sharing and W_out scale are unmeasured. A rank-2 solution exists and is held: a search deficit.
metadata:
  type: project
---

**RANK 3, 2026-09-28 (`RANK3_RESULTS.md`): rank 3 per head sits with rank 4.** Per-head r=3 (built
from our r=2's base, T=1024, 900 ep) solves 6/8, acc 0.987 (per-head r=2 2/8, 0.885; per-head r=4 8/8,
0.999). 3 - 2: +0.102, perm p 0.027, Holm 0.054, solved Fisher 0.13 -- registered RANK 3 SUFFICES, at
the registered edge on every count (accuracy only, Holm just above .05, exactly 6/8). 4 - 3 UNMEASURED
(+0.012). The two unsolved r=3 seeds sit in the non-cancelling basin (opposition 1.58 / 1.83): rank 3
makes the basin rarer, it does not remove it. Favours "rank 2 is special" (= the torus's 2 DOF); the
untested prediction is a 3D torus where rank 3 fails.

**NOT RESCUED BY 2x BUDGET, 2026-09-29 (`LOOP_RANK_E1800_P1_RESULTS.md`, H1 part 1).** From scratch at
1800 epochs: shared r=2 0/8 (0.894 -> 0.908), shared r=4 7/8 (Fisher p 0.0014). Not "never": 5/8 rank-2
runs still descending. H1 part 2 (loop and 4 layers at 1800 ep) deferred.

**H1, 2026-09-27 (`LOOP_RANK_RESULTS.md`): search aids partly recover rank 2, registered verdict
UNMEASURED.** r=2 + loop x4 (identical params/init to r=2) 2/8 solved, acc 0.973; r=2 at 4 real
layers 5/8, 0.990; r=2 0/8, 0.894; r=4 8/8, 0.998. Accuracy fires for both aids; solved count only
for depth. Three loss regimes -- the aids leave r=2's and never reach r=4's. The 0.05 cutoff sits
inside the aids' spread (at 0.08+ H1 would have passed), so uncertain, not negative. Depth beats the
matched-param loop, so "search at constant capacity" is NOT shown.

**RESOLVED 2026-09-25 (`RANK_SEP_RESULTS.md`): it is the PER-HEAD rank.** Five arms from our r=2's
initial weights, differing only in the bottleneck; SOLVED within 900 ep at T=1024: per-head rank 2 ->
0/8 (shared r=2), 2/8 (per-head r=2), 2/8 (block-diagonal r=4); per-head rank 4 -> 8/8 (shared r=4),
8/8 (per-head r=4). Total latent size does not track it. Separated: per-head rank FIRES (D-C_bd, both
block-diagonal, Fisher/perm p 0.0070); SHARING unmeasured (D-C); W_out per-entry scale unmeasured
(C_bd-B); initial angle scale excluded (A and B fail at normal scale). Supersedes the "unseparated"
caveat below. Still SEARCH, not capacity (the projection exists and is held).

**STATE, 2026-09-24 (supersedes the older paragraphs below where they conflict).**
- **RESOLVED at matched initialisation** (`RANK_MI_RESULTS.md`, 2026-09-24 22:47): every arm built
  from our r=2's initial weights; within 900 epochs at T=1024 our shared r=2 solves 0/8, the paper's
  per-head r=2 2/8, shared r=4 8/8 (C-A Fisher p 0.0002, C-B 0.007; acc 0.894 / 0.885 / 0.998). B and
  C have the same 4 latent dims and identical initial W_in; per-head rank, cross-head sharing and W_out
  per-entry scale are UNSEPARATED (corrected after review; initial angle scale is matched, 0.335/0.343/0.344).
  The stored r=4's 8/8 was not an init artifact. B-A unmeasured. Separating arms: C_bd, per-head r=4.
  Reproduction of stored r=2 seed 3: bit-exact over 900 epochs.
- **The old win is out of distribution only.** Trained T=128: r=2 0.993 vs r=4 1.000 at T=128; the
  gap appears at T=512 (+0.038) and T=1024 (+0.085, `RANK_SWEEP.md`). 94% of the +0.085 is short-gap
  (<128 step) revisits late in the sequence (r=2 0.903 vs r=4 0.999): in-distribution lag, unseen
  absolute position. Old losses did not overlap (r=2 0.0011-0.0874, r=4 <= 0.0006).
- **Matched length** (`RANK_MATCHED_RESULTS.md`, `runs/rank_matched_e900`, trained and tested at
  T=1024, 900 ep, 8 seeds): r=4 SOLVED 8/8, r=2 0/8 (4 stalled at loss 0.40-0.68, 4 descending at
  0.05-0.32); acc 0.997 vs 0.894, +0.103 (perm p 0.0003); Fisher p 0.0002 on solved counts.
  **Registered verdict UNREADABLE** (more than 2 runs per arm still descending). The 900-epoch
  warm-restart continuation (`runs/rank_matched_e900c`): r=4 7/8 solved, r=2 1/8, UNREADABLE again;
  by Amendment 3 no further extension. Under a restart the classes measure recovery from the kick:
  the four r=2 runs called STALLED at 900 all improved in cycle 2.
- **r=2 can represent the solution.** Projecting each solved r=4 onto the top two directions of its
  action latent (top-2 energy >= 0.9995) and freezing it scores **0.995 at T=1024 on 8/8 seeds**
  (0.957 at T=2048; `RANK_PROJ_FROZEN.md`, untracked at the time of writing). So the gap is SEARCH,
  not capacity -- and not a skewed basis: within r=2, skew does not predict accuracy (r = -0.35/+0.30),
  so "skew is the mechanism" is withdrawn (skew is the SYMPTOM: from-scratch r=2 stalls at opposition
  0.87-1.78 on 7/8 seeds). **Warm-start stability test, S1 STABLE** (`RANK_PROJ_RESULTS.md`): r=2 started
  from the projection and trained with the continuation recipe solves 7/8 = the r=4 control's 7/8; its
  code stays cancelling through the restart kick (reviewed with snapshots). Exists, stable, not found =
  search. Untested: whether r=2 can LEAVE the non-cancelling configurations it stalls in.
- **Our bottleneck differs from the paper's.** Ours shares one r-dim latent across heads; the paper's
  `W_in` is per head (`R^{d x nh x r}`, `papers/txt/mapformer.txt` ~l.1512). At 2 heads our r=2 has half
  the paper's latent dims (ours r=2 < paper r=2 < ours r=4: 2 / 4 / 4 dims, 384 / 640 / 768 params), so
  "use r=4" may only restore the paper's latent-dimension count. Also unseparated: `w_out`'s init bound
  1/sqrt(r) makes Adam's relative step ~1.4x smaller at r=2. Next: per-head r=2 + r=2 with r=4's init
  scale, one batch (reviewer's plan, ~10 GPU-h). Say so wherever r is compared with the paper.
- **Scope:** MapWM family only (MapPoPE r=4 +0.019, unmeasured, `MAPPOPE_R4_RESULTS.md`); on Bach at a
  512 context the order INVERTS (r=1 best, below). Until the continuation lands, "use r=4" is a
  recommendation about what training FINDS at our shared-bottleneck r=2, not about capacity.

The paragraphs below are the 2026-09-04..17 record; read them through the block above.

**[OOD-only -- see the top block] r=4 IS THE DEFAULT TO USE (2026-09-04, RANK_SWEEP.md).** Torus, 8 seeds, one
batch: against r=2 at T=1024, r4 **+0.085** (t 3.57, 8/8, sign p=0.008), r8 +0.091,
r16 +0.079, r32 +0.095. A **STEP at r=2, not a slope** -- flat from r=4 up. Costs
+384 params (+0.19%) and also cuts seed sd 0.064 -> 0.012 at T=1024, which matters
because sd sets every MDE in this project.

**[WITHDRAWN as the mechanism, 2026-09-24 -- see the top block] WHY: the r=2 code is SKEWED, not too small** (ACTION_GEOMETRY.md). Given 4 dims
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
  tested: Fig. 4 is an EM model (App. C.3 is explicitly about EM's separate pools),
  and MapWM has no such split (it has no position-only stream -- NOT because it is additive; it isn't). See run_em_fig4.sh.

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

**CONFOUND found 2026-09-05, and it invalidates the per-knob attribution.** The
Selective RoPE "single-knob" arms are NOT single-knob. `SelectiveAngle` has no
omega -- its readout is `A = tau*I` with one scalar -- so every arm also deletes
`path_integrator.omega` and `action_to_lie`, swapping
`A: diag(omega) W_out -> tau I` and discarding 64 learnable
geometrically-initialised frequencies. **The conv / rank / gate rows in
SELECTIVE_ROPE.md cannot attribute effects to conv, rank or gate.** The sign flip
between tasks survives as an observation about the arms as built. RANK_SWEEP is
unaffected -- its arms differ only in `bottleneck_r`.

**Prior art, 2026-09-06:** the r=4 result stands but the FRAME around it does not
-- see [[reference-positional-landscape]]. Rank is one of only two things left.

**CONFLICT (2026-09-17, `JSB_LENGTH_RESULTS_RANK.md`):** on Bach Chorales trained at a 512-token
context, the ordering INVERTS -- MapWM r=1 is the best arm at every position bucket (0.912 NLL at
2-4x beyond the context) against r=2's 1.397 and r=4's 1.660, detectable and growing with length.
"Use r=4" remains a torus-navigation result; it is not a general rule. Suspected difference: the
torus needs a well-conditioned 2-D displacement basis, serialised music has no displacement to
represent and a wider bottleneck mainly lets more content drive the phase. Unresolved.

