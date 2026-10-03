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
