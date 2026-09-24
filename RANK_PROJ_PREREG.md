# Warm-start stability test: does r=2 HOLD the solution once it has it? (2026-09-24, pre-registered)

Companion to `RANK_MATCHED_PREREG.md`. Trained from scratch at T=1024 for 900 epochs, r=2
solved 0/8 runs and r=4 8/8. A rank-2 projection of a trained r=4 model keeps its accuracy
(`probe_rank_projection.py`: 0.9995 on seed 6), so a rank-2 solution EXISTS. Rule 32:
existence, then stability -- warm-start the solution twice, frozen and trainable.

## Construction (`build_rank_proj.py`, run before this file was written)

For each seed s = 0..7, the solved T=1024 r=4 checkpoint (`runs/rank_matched_e900`) is
projected onto the top two directions of its action latent over the vocabulary; every
other weight is kept. Top-2 spectral energy 0.99952-0.99994, relative error in the
per-token angle increments 0.0015-0.0064. Written to `runs/rank_proj/p0/Vanilla_s<s>`.

## Arms

- **FROZEN**: the projected r=2 models, evaluated as they are. Existence on all 8 seeds.
- **TRAINABLE** (`runs/rank_proj_train`): each projected model trained with EXACTLY the
  continuation's recipe (`RANK_MATCHED_PREREG.md` Amendment 3): 900 epochs, 5% warmup to
  lr 1e-3, cosine to 1e-4, fresh AdamW, seed s, `--data-seed-offset 1`. It therefore sees
  the SAME walk stream as the continuation's r=4 seed s.
- **Control**: the continuation's r=4 arm (`runs/rank_matched_e900c`), which re-trains the
  SAME solved r=4 weights with the same recipe, seed and data. Only the rank differs. The
  restart perturbs a solved model (smoke test: loss 0.004 -> 0.27 -> 0.013), so the
  question is whether the model RECOVERS the solution at rank 2 as it does at rank 4.
- Reference: the continuation's r=2 arm (same recipe, started from r=2's own unsolved
  900-epoch weights) -- the from-scratch lineage.

## Readouts

Run classes as in Amendment 2 (SOLVED final-5% loss < 0.05; STALLED; DESCENDING), on each
run's own 900 epochs. Primary: SOLVED count, TRAINABLE vs control, Fisher exact. Secondary:
T=1024 accuracy (exact permutation test) and strata; the peak loss during the restart and
the epoch it recovers below 0.05; action-code opposition (does the trained r=2 keep a
cancelling code?); FROZEN accuracy against the source r=4 at T=1024.

## Branches (in order)

1. **Uninformative** if the control re-solves on fewer than 6/8: the restart is too harsh
   for either rank; go to the gentle version below.
2. **S1 STABLE**: TRAINABLE SOLVED on >= 6/8. Rank 2 holds the solution through training
   and recovers it after the kick, so r=2's from-scratch failure is SEARCH: the solution
   exists and is an attractor once reached, but training does not reach it.
3. **S2 UNSTABLE**: TRAINABLE SOLVED on <= 2/8 (control >= 6/8). Started inside the
   solution, rank 2 does not keep or re-find it under this recipe. Next: the gentle
   version, to split "the kick ejects it" from "the landscape does not hold it".
4. **MIXED** otherwise: report counts and Fisher; no mechanism sentence.

Gentle version (not launched; only if branch 1 or 3 fires): same arms at peak lr 1e-4
for 300 epochs, with the matching r=4 control.
