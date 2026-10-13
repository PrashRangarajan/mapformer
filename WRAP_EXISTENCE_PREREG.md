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

## Amendment 1 (2026-10-13, after the independent check `docs/audits/2026-10-13/wrap_existence_audit.md` and the pilot; before launch)
Pilot (seed 690, outside the batch; reported, never pooled): SO SOLVED (acc 1.000); AO from scratch 0.809. The registered
min-norm construction was exact frozen (1.000 = 1.000) but its W_out was 1825 vs the signed 0.39 (steps up to 1199), and its
continuation lost the solution in the warm-up (loss 0.05 -> 1.05 by epoch 10) while SO_cont stayed at ~0.002; a per-channel
gauge version (W_out 8.4) lost it faster. The check showed why: in a solved signed model the two action latents are nearly
antiparallel (cos -0.993 on the pilot, -0.996..-0.999 on RING_DIAL), so keeping W_in forces an ill-conditioned W_out, and
Adam's per-entry step then moves the phase far (rule 15). With weight decay 0 it collapses too. **A NOT HELD under those
constructions would be a parameterisation artefact.** Changes:
- **A1 (primary construction = balanced).** W_in acts only on the two action embeddings (observation steps are 0), so it is a
  free gauge: re-solve W_in so the action latents are c e1 and c e2 (c = the source's mean action-latent norm), set
  omega' = max(|omega|, 1) per channel and steps remainder(omega Delta, 2 pi) / omega' in [0, 2 pi); W_out[:, :2] = steps^T / c.
  Exact (pilot: angle error < 1e-6 rad, frozen 1.000, max |W_out| 1.21, |W_in| 0.27 vs 0.20; held 1.000 for 400 CPU AdamW
  steps in the check). **New arms: AO_bal** (this construction) and its matched control **SO_bal** (SignedAO_r4 at the same
  W_in and omega' with the UNWRAPPED signed angle, steps omega Delta / omega'), built by the same function, so AO_bal and
  SO_bal differ only in the wrap (same torch RNG consumption, same data stream). AO_wrap (min-norm) and AO_wrapg (per-channel
  gauge) stay as DESCRIPTIVE arms against SO_cont. 7 arms x 8 seeds = 56 runs.
- **A2 (construction fix).** Wrap in step space, remainder(Delta, 2 pi / |omega|): trained omega crosses 0 (pilot -0.012;
  RING_DIAL s31-33 too), and the original remainder(omega Delta, 2 pi) / omega was wrong for omega < 0 (synthetic: 2.46 rad).
- **A3 (E1 per construction)** and E2 gated on that construction's E1 EXISTS (and its control's, for SO_bal). E1 also records
  the max logit difference (8 trajectories at T = 128 and 512), descriptive.
- **A4 (E2 per construction; LOST).** PRIMARY: AO_bal vs SO_bal. A run is LOST if its max per-epoch training loss during the
  continuation is > 0.3, or > 10x its control's max at that seed AND > 0.05. HELD = SOLVED on >= n_s - 1 AND mean accuracy
  >= control - 0.01 AND LOST on <= 1 seed. NOT HELD = Fisher (SOLVED) p < .05 with the control ahead, OR sign-flip p < .05
  with d <= -0.05, OR Fisher (not LOST) p < .05 with more LOST in the constructed arm. PARTIAL otherwise. Reachability at
  small n_s: the sign-flip branch cannot fire below n_s = 6 (min p 0.125 at 4, 0.0625 at 5); Fisher needs 0/4 vs 4/4 at
  n_s = 4.
- **A5 (E3 gate).** E3 is UNREADABLE if SO SOLVED < 4 (the smoke read NO SIGN COST with both arms at the floor).
- **A6 (secondaries).** wrap_share selects channels by |wrap(angle(+1))| > 0.05 (wrap-invariant); the max wrap-aware change
  of the action angles from construction to the end of training; frozen sensitivity of SO_bal vs AO_bal to omega or W_out x
  (1 + eps N(0,1)), eps 1e-3 / 1e-2, 3 draws, 200 trajectories (is the wrapped solution more fragile at a matched gauge?).
- **A7 (driver).** GUARD adds train_variant.py and lie_groups.py; the artifact count is counted (112), not asserted by
  construction; 5 jobs per GPU; code committed before launch.
- **Meaning, stated now (from the check):** HELD under an exact construction shows that a stable non-negative solution
  exists, so the RING_DIAL deficit is search. NOT HELD under the balanced construction, with SO_bal held, means "unstable at
  the scale training uses". NOT HELD only under an ill-conditioned construction (AO_wrap, AO_wrapg) means nothing about
  the class.
- Disclosed, not fixed: optimizer state is fresh in every continuation; SO_cont builds no extra model, so its dropout RNG
  stream differs from the constructed arms'. The lock fd is inherited by children (a relaunch after a failure is refused
  until jobs end).
Smoke (CPU, 2-epoch runs of all 7 arms on 4 seeds from the pilot SO): every construction exact (angle error <= 3e-6 rad,
frozen = source), every branch executes. Cost: 56 runs at ~10-13 min, 10 at a time on both GPUs: ~75 min.
