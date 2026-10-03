# Leak remedies on the new-object task -- pre-registration (2026-10-03, before any run)

## Question
`docs/WHAT_WHERE_CHECKS.md` section 3 (eval-only, n=1): in path-integrating models, object ("what") tokens leak
into the step ("where"): zeroing object-token steps lifts unseen-object accuracy 0.990-0.993 -> 0.9996-0.9999, and
scaling the object codes on the embedding side x2 / x4 costs 0.04 / 0.11-0.14 with the content channel unchanged.
Mechanism: the step reads the raw embedding, before LayerNorm. Do two remedies remove the leak, and at what cost?

## Task, recipe (`environment_newobj.py`, `model_codes.py`, `train_newobj.py`; gated `docs/audits/2026-09-27/gate_newobj.py`)
2D torus 32x32 walk (GridWorldND), fresh map and 16 fresh objects per sequence, objects = fixed random 64-d codes
through a learned encoder/readout, train pool vs disjoint test pool. T = 1024 steps, batch 16, 900 epochs x 98
batches, lr 1e-3, warmup + cosine, 1 layer, 2 heads, d 128, `--data-workers 3`, pool 1000. Seeds 0-7 (the pilot used
seed 100). One batch, every arm retrained.

## Arms (all MapWM r=4 bases; every base weight identical across arms at a seed -- gated)
| arm | step | role |
|---|---|---|
| MapWM | W_out W_in emb(x) | baseline |
| ActOnly | as MapWM, times 1[x is an action] | TEM-t's action-only update; reference (told which tokens are actions; leak 0 by construction) |
| NormStep | W_out W_in LN(emb(x)) | remedy that keeps the step learned for every token, removing only its norm sensitivity |
Gate (2026-10-03): ActOnly's object and blank steps are exactly 0; NormStep adds only `step_ln`; causal leak 0.

## Readouts (`leak_eval.py`, validated: reproduces the pilot's E0 numbers exactly; test pool, 200 eval sequences)
Object-identity accuracy with the object codes scaled on the embedding side by s in {1, 2, 4}, intact and with
object-token steps zeroed; leak L(s) = zeroed - intact. Train-pool accuracy at s=1 as a reference.

## Hypotheses (per remedy arm R in {ActOnly, NormStep}; exact permutation test, 8 vs 8, two-sided, stats_core)
- Robustness (x4 is a distribution shift, rule 10): acc_R(x4) - acc_MapWM(x4).
- Cost in distribution: acc_R(x1) - acc_MapWM(x1).
Branches, fixed now, per arm:
- **REMEDY** -- the x4 contrast fires (p < 0.05, R higher, d >= 0.02) AND median L_R(x4) <= 0.01 AND the x1 contrast
  does NOT show a significant loss (not: p < 0.05 with d <= -0.01).
- **REMEDY WITH A COST** -- the x4 conditions hold but the x1 contrast shows a significant loss of >= 0.01.
- **NO REMEDY** -- the x4 contrast does not fire and median L_R(x4) > 0.05.
- Otherwise: reported as it falls.
Secondary (no verdict): L(x1) and L(x2) per arm; x2 contrast; train-pool accuracy; run classes (SOLVED / STALLED /
DESCENDING) and final loss; ActOnly vs NormStep.
Void: any of the 24 runs missing; md5 guard trips; ActOnly's L(s) differs from 0 by more than 0.002 (wiring).
Scope: one task, 1 layer, r=4, T=1024, 900 epochs, n=8. Cost ~5-6 h (8 concurrent).

---

## Amendment 1 (2026-10-03 ~13:00, after an independent audit, BEFORE any result was read)
No code bug. But the registered x4 test **cannot fail for either remedy arm**: scaling the object codes on the
embedding side changes ActOnly's and NormStep's scored logits by <= 4e-6 (ActOnly has no object step; NormStep's
step reads LayerNorm(emb); every arm's content path is pre-LN), so acc_R(x4) = acc_R(x1) and the x4 contrast reduces
to acc_R(x1) - acc_MapWM(x4), which fires for any remedy arm >= ~0.88 at x1. The registered branches are still
computed and reported as registered, but are read as: "the remedy trains to within 0.01 of MapWM at x1 and its
in-distribution leak is <= 0.01" -- robustness to a norm shift is guaranteed by construction. The x2/x4 rows for
ActOnly and NormStep are CHECKS of the construction, not tests. Added as declared secondaries
(`analyze_leak_secondary.py`, no verdict):
- (a) the x1 contrast R - MapWM with its exact-t MDE (if the leak is removed at no cost the expected value is about
  +L_MapWM(x1) ~ +0.007, likely below the MDE);
- (b) L_NormStep(x1) vs L_MapWM(x1) (permutation): does a norm-invariant step leak less IN distribution -- the only
  real question about NormStep;
- (c) step gain per arm: rms(object step) / rms(action step), and the blank-step gain;
- (d) budget (rules 2-4): the pilot path arms were DESCENDING at 900 epochs (tails 0.07-0.08), and both remedies change
  the optimisation (rule 15): final loss beside every x1 number and r(final loss, x1 accuracy) over all 24 runs;
  every number is scoped to 900 epochs;
- (e) MapWM's eval-only blank-step zeroing on the batch checkpoints: ActOnly - MapWM bundles no object leak, no blank
  step and an oracle action label, so an x1 difference cannot be attributed to one of them.
Also: the leak readout is E0's L_obj_only (object steps only), not E0's field L; the void check is printed, not
enforced -- read the last line of LEAK_ANALYSIS.txt; before reading, re-run `md5sum -c runs/leak/code_md5.txt`.
Directional shifts that LayerNorm cannot remove (E0's random-map / projection conditions per arm) are deferred.
