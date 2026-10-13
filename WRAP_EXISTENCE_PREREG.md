# Can a non-negative rotary step hold the ring map? (rule 14 existence) -- pre-registration (2026-10-13, before any batch run)

## Question
RING_DIAL (`RING_DIAL_RESULTS.md`): on the 1D ring at p_plus 0.5, a non-negative rotary step (Abs_r4, |W_out W_in x|) learns
the map on 2/8 seeds vs 8/8 for the signed step; 3x the budget does not help (`RING_BUDGET_RESULTS.md`, 3/8 vs 3/8); no
non-negative run finds the cancelling code mod 2 pi. We read this as LEARNABILITY because a non-negative rotary step can
REPRESENT the signed code (angles wrap mod 2 pi: a step 2 pi / omega - Delta reproduces -Delta). Rule 14 says that reading
needs the existence shown in the class at a trained scale: **construct a non-negative solution, check it frozen, then check
whether training keeps it.** If training does not keep it, the solution is not a stable point of the optimiser, a stronger
statement than "hard to find".

The first attempt (`docs/audits/2026-10-13/wrap_existence_build.py`, CPU): the wrap is exact for the action angles (error
<= 5e-3 rad), but trained Signed_r4 ring models rely on their OBSERVATION steps (zeroed: accuracy 1.000 -> 0.47-0.96 on all
8 seeds), and the rank-4 bottleneck cannot wrap the observation steps as well. So this test removes observation steps from
the class by construction, in both signs.

## Models (`model_wrap_exist.py`; trainer `train_wrap_exist.py`)
- **SignedAO_r4**: Signed_r4 (MapWM, shared rank 4) with every non-action token's step 0 by construction; the +1 and -1 steps
  are W_out W_in e(a), free (RANK_TESTS' R1AO at shared rank 4).
- **AbsAO_r4**: the same with |.| per channel. At one seed both start from identical weights.
- **Wrap construction** (`wrap_construct`): from a trained SignedAO_r4, an AbsAO_r4 with W_in and every other weight copied
  and W_out the minimum-norm solution of latent(a) . W_out[c] = (omega_c Delta_c(a) mod 2 pi) / omega_c. Steps are then >= 0,
  |.| is the identity on them, and every rotary angle equals the signed model's mod 2 pi (smoke: logits agree to 5e-6).

## Arms (ring 32 cells, p_plus 0.5, RING_DIAL recipe: T = 128, batch 128, 98 batches, lr 1e-3, cosine with 5% warmup, 1
layer, 2 heads, d 128, 300 epochs; seeds 600-607, fresh; n = 8)
| arm | what | init |
|---|---|---|
| SO | SignedAO_r4 from scratch | seed |
| AO | AbsAO_r4 from scratch | seed (same weights as SO) |
| SO_cont | SO's checkpoint trained 300 more epochs | SO at that seed |
| AO_wrap | AbsAO_r4 constructed from SO's checkpoint, evaluated frozen, then trained 300 epochs | wrap_construct(SO) |
SO_cont and AO_wrap train on the same fresh walk stream (data_seed_offset 10^6) with the same schedule (warmup to lr 1e-3,
cosine), so SO_cont controls for the re-warming of the learning rate. Weight decay 0.05 applies as in training (it pulls the
wrapped W_out, which is larger: reported). Evaluation: `train_cancel.evaluate`, a fresh stream (np seed 10^6, 400
trajectories, maps redrawn per trajectory), eval mode, T = 128 (primary) and 512. SOLVED = `stats_core.classify_run` on the
arm's own training losses (final 5% < 0.05). Floor (CANCEL_RESULTS): 0.554 at p_plus 0.5.

## Registered decisions (`analyze_wrap_existence.py`)
- **P0 (prerequisite): SO SOLVED count.** E1 and E2 are read only on seeds whose SO run SOLVED (n_s). If n_s < 4, E2 is
  reported as UNREADABLE (too few sources), not as a result.
- **E1 (existence, frozen): EXISTS** if on every SOLVED source the constructed AO_wrap model's frozen accuracy is within 0.002
  of the source's (same stream) and its max action-angle error is < 1e-2 rad. Otherwise CONSTRUCTION FAILED (a bug to fix, not
  a finding).
- **E2 (held under training; the primary), on the n_s solved sources:** compare AO_wrap with SO_cont after 300 epochs.
  - **HELD**: AO_wrap SOLVED on >= n_s - 1 AND mean accuracy (T = 128) >= mean SO_cont - 0.01.
  - **NOT HELD**: Fisher (SOLVED) p < .05 with SO_cont ahead, OR paired sign-flip p < .05 with mean AO_wrap - SO_cont <= -0.05.
  - **PARTIAL** otherwise (reported as it falls, with the per-seed table).
- **E3 (learnability without observation steps): AO vs SO from scratch** (n = 8 each; unpaired permutation test on accuracy, as RING_DIAL).
  - **SIGNED LEARNS BETTER**: Fisher (SOLVED) p < .05 with SO ahead, OR perm p < .05 with mean AO - SO <= -0.05.
  - **NO SIGN COST**: AO SOLVED >= SO SOLVED - 1 AND mean AO >= mean SO - 0.01.
  - **UNMEASURED** otherwise.
E1-E3 are separate questions; no multiplicity correction.

Secondaries (declared, no verdict): wrap share (channels with |omega step(+1)| > 0.05 whose |wrap(angle(+1) + angle(-1))|
< 0.1 rad) for every arm, and for AO_wrap at construction vs after training (does training move it off the wrap?); the
accuracy trajectory is not logged, so "held" is read at the end only; accuracy at T = 512 (OOD, rule 10); the 1/(1-p)
dropout re-score (one layer); |W_out| max and step max, signed vs wrapped; paired loss AUC (AO - SO, scratch); cross-batch and
labelled: AO here vs RING_DIAL's Abs_r4 (2/8) -- did removing observation steps help the non-negative step?

## What each outcome means (stated now)
- E1 EXISTS + E2 HELD: the non-negative rotary step can carry the map and training keeps it; the RING_DIAL deficit is search
  (finding the wrap), as for rank 2 on the torus.
- E1 EXISTS + E2 NOT HELD: the wrapped solution is not stable under this optimiser (the wrapped steps are 2 pi / omega - |Delta|,
  large on low-frequency channels, so small relative changes in omega or W_out move the phase far): learnability in the
  stronger sense, at the scale training produces.
- E3 NO SIGN COST would mean RING_DIAL's deficit was about observation steps, not sign; SIGNED LEARNS BETTER replicates the
  sign cost in a class where only the actions move.

## Process and cost
Pilot on seed 690 (outside the batch): SO, AO, then SO_cont and AO_wrap (the pilot checks that SO solves, the construction
is exact at a trained scale and the timing; it is reported, never pooled). Independent code check (rule 29) before launch;
its findings become Amendment 1. 32 runs x ~10 min at 6 per GPU on both GPUs (if free): phase 1 (SO, AO; 16 runs) then
phase 2 (SO_cont, AO_wrap; 16 runs), ~1 h.
