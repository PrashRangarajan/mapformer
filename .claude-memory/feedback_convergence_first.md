---
name: feedback-convergence-first
description: Convergence, recipe, floor, power and loss overlap BEFORE any architectural comparison (CLAUDE.md rules 1-5, 12). The lm200 leaderboard and four August claims died of this.
metadata:
  type: feedback
---

The why behind CLAUDE.md rules 1-5 and 12. `python3 -m mapformer.experiment_audit --runs-dir <dir>
--control <inert-twin> --control-of <real-arm>` checks most of it in ~30 s, but its flat-slope
convergence test is the kind rule 3 distrusts (it passed stalled and still-descending runs). Where
they apply, use Amendment 2's SOLVED / STALLED / DESCENDING classes (`RANK_MATCHED_PREREG.md`).

**Why:** ~200 arms in 2026-08-26..28 produced 5 architectural claims and 4 were retracted, each with
the same root cause, rediscovered days apart.

1. **Convergence + LR schedule.** `LinearLR(1.0->0.0)` from step one has no warmup; on a
   plateau-then-cliff landscape a run cannot escape late, so the budget measures "did the transition
   fire early". 5% warmup + cosine-to-10% moved an arm **0.448 -> 0.990 on the same task** and
   INVERTED a headline.
2. **Measure the noise floor** with an arm provably function-identical to a real one (params
   identical, effect multiplied out, zero grad): 0.150 on MiniWorld, larger than most effects chased.
3. **Accuracy may just be the training loss** (r = -0.996 over 57 runs); then the eval adds nothing
   and only loss-matched residuals mean anything.
4. **"Null" requires power.** Say "unmeasured" below the MDE. Conditioning on convergence can select
   into a ceiling and show ~0 by construction; report threshold sensitivity.
5. **Seed the comparison you claim.** Seeds on "A vs baseline" do not support "A vs its components".
6. **Cross-check your own gate data against the mechanism.** The attention-horizon story was
   falsified by gate G6, collected before training and never looked at.

## A leaderboard that tracks final loss is measuring optimisation (lm200, retracted 2026-07-16)

April lm200 checkpoints never converged (final CE ~1.0, not ~0.005); May ones did. The ranking was
monotone in final loss: Vanilla 0.716 (loss 1.22), Level15 0.819 (1.01), NoDrop 0.948 (0.24), GSF
0.956 (0.0007), TEMFaithful 0.969 (0.0004). Retrained, Level15 reaches 0.996 and beats TEM (0.982).
Dead with it: "TEM leads lm200", "removing post-attention dropout buys +13pp", "GSF closes 95% of
the TEM gap", "NoDrop and GSF are accuracy-substitutes but NLL-complements". The dropout-hurts-rare-
retrieval story may be true; it has no surviving evidence. Scope is narrow: clean and noise
checkpoints retrain bit-identically; the cause is the landmark-cell RNG, which runs only when
`n_landmarks > 0`. Check r(final loss, acc) before reading any ranking (it has reached -0.999).
Never compare to a stored checkpoint (rule 12).

## Check the recipe before believing a ceiling (compositional, 2026-09-08, `COMP_HEADROOM.md`)

The task sat at cross_nb 0.415 (floor 0.072) for months. Every number used LinearLR-from-step-one at
lr 3e-4, 50 ep; the trainer had no `--schedule` flag. Published recipe 0.354 vs cosine / 1e-3 /
150 ep **0.514: +0.160, 7/8**, larger than any architecture effect on that task (hierarchy +0.13).
`grep -l LinearLR train_*.py` finds trainers that predate the fix. My registered prediction that the
recipe would compress variance (it cut sd 3.5x on the torus) failed: sd 0.070 -> 0.151, so the
mechanism is unidentified. Ask every time: is the effect I cite larger than the recipe effect on the
same task? The hierarchy claim was re-measured because of this ([[project-hierarchy-negative]]).

## Loss-matching needs overlapping losses (2026-09-13/14)

Twice a loss-matched contrast removed the effect by construction: MONOTONE Q1 (losses 0.0003 vs
0.31) and PAPER2X2 position (index 0.68-0.96, path 0.00-0.38). When the manipulation causes the loss
gap, the fit is identified by the arm difference and the residual is uninformative, not null. Check
that loss varies within arms and overlaps across them BEFORE registering a loss-matched verdict;
otherwise the raw contrast is primary. State the pool: intermediate-loss arms change the verdict.
**At matched length (train = test) a near-zero loss-matched residual is predicted whatever the
mechanism**, which is why "the r=4 gap is training speed" was withdrawn (`RANK_MATCHED_RESULTS.md`).

## Floor and control convergence (2026-09-20, Dyck/PoPE line)

Report the task's measured floor beside every headline: two batches were read entirely below a
no-stack n-gram floor, and a conclusion drawn from sub-floor F1 was withdrawn. An effect measured
against a control that TRAINS BETTER is not attributable; check loss overlap before the batch. In
`CROSS_RESULTS.md` accuracy tracks final loss at r = -0.95 to -0.99 with non-overlapping losses, so
the surviving effect cannot be loss-matched at all.

Related: [[project-miniworld-flip-negative]], [[feedback-validate-task-first]].
