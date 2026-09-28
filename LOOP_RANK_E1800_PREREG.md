# H1 at a doubled budget -- pre-registration (2026-09-27, before any run)

## Why
`LOOP_RANK_RESULTS.md` (900 epochs): r=2 + loop x4 (L) solves 2/8, r=2 at 4 real layers (L4) 5/8,
plain r=2 (A) 0/8, r=4 (C) 8/8. The registered SOLVED cutoff (final-5% loss < 0.05) falls inside
the aids' loss spread (L 0.045-0.127, L4 0.011-0.070), several L/L4 runs were still descending,
and at a cutoff of 0.08 H1 would have passed. The verdict was UNMEASURED. Rule 4: extend the
budget rather than argue about the cutoff. A warm-restart continuation is NOT used: in
`RANK_MATCHED_RESULTS.md` (e900c) the restart knocked every r=4 run up and the classes then
measured recovery from the kick. Every arm is retrained FROM SCRATCH with a 1800-epoch cosine
schedule, in one batch.

## Arms (torus, trained AND tested at T=1024, 1800 epochs, 8 seeds, one batch)
| arm | variant | params | passes/step |
|---|---|---|---|
| A r=2, 1 layer | `Vanilla` | 204,373 | 1 |
| L r=2 + loop x4 | `Looped` | 204,373 | 4 |
| L4 r=2, 4 real layers | `Vanilla_L4 --n-layers 4` | 799,189 | 4 |
| C r=4 (matched init) | `Vanilla_r4mi` | 204,757 | 1 |
Recipe otherwise identical to `runs/loop_rank` (batch 16, lr 1e-3, warmup + cosine, 98 batches/epoch,
`--data-workers 3`, `--save-full-state`, explicit attention path -- `--fast-attn --deterministic`
was timed at only 1.25x on `Looped` and is not worth a new series).

## Readouts (SOLVED = final-5% loss < 0.05, the registered cutoff, unchanged)
Co-primaries, L vs A: SOLVED count (Fisher) and T=1024 accuracy (exact permutation), `stats_core`.
Secondary: L4 vs A, L vs L4 (Holm over the three); C vs L and C vs L4 (does rank 4 stay in a class
of its own); final-loss regimes per arm (the three-regime table); T=512 / 2048; strata; sensitivity
of the solved counts to the cutoff (0.02 / 0.05 / 0.08) reported as in `LOOP_RANK_RESULTS.md`.

## Branches, fixed now (read in this order)
- **RANK 2 WAS BUDGET** -- A itself solves >= 6/8. The 900-epoch rank-2 deficit was a budget
  effect; H1's question dissolves, and the rank line's citable table must be re-scoped to 900 epochs.
- **H1 CONFIRMED** -- L vs A fires (either co-primary, p < 0.05) with L >= 4/8 SOLVED.
- **H1 REFUTED** -- L within MDE of A on accuracy, L vs A does not fire, and L <= 2/8.
- **UNMEASURED** otherwise.
- **Regime reading** (secondary): if any L or L4 run's final loss falls inside C's range at 1800
  epochs, "extra search leaves r=2's regime but never enters r=4's" is withdrawn.
Void: any of the 32 runs missing; md5 guard trips.
Scope: torus, T=1024, n_heads 2, d 128, one recipe, 1800-epoch budget, one loop count (4).
Cost: L and L4 ~7.4 s/epoch solo (~3.7 h solo, ~5 h shared); A and C ~2 s/epoch (~1-1.5 h).
~100 slot-hours, ~25 h wall at 2 jobs/GPU. Queued behind the rank-3 and sign batches.
