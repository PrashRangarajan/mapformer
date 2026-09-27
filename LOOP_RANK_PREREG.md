# H1: if rank 2's deficit is SEARCH, a search aid should fix it (2026-09-26, before any run)

## The hypothesis, and why it is predicted rather than fished

`RANK_SEP_RESULTS.md`: on the torus at T=1024, every arm whose per-head content-to-angle map has
rank 2 finds the solution on 0-2 of 8 seeds, every rank-4 arm on 8/8. `RANK_PROJ_RESULTS.md`: a
rank-2 solution EXISTS (0.9955) and is HELD under training. So rank 2's failure is search, not
capacity. If that is right, a mechanism that improves SEARCH without adding capacity should
recover it.

The loop is that mechanism, and the project has independent evidence it acts on search rather than
expressivity: it is parameter-identical to 1 layer, its contribution is to the FLOOR not the
ceiling (`LOOP_HEADROOM.md`: 8/8 seeds >= 0.77 where the baseline spans 0.11-0.80), it gives
+0.346 unpaired on Match-Query (`REFINE_RESULTS.md`), and its advantage SHRINKS when the baseline
is already trained well (`MQ_NOISE_2X2_C2.md`, `L15_LOOP_2X2.md`). Here the baseline is as badly
off as it gets (0/8), so the loop should have maximal room.

**This is a positive prediction with a named mechanism, registered before the runs.**

## Arms (torus, trained AND tested at T=1024, 900 epochs, 8 seeds, one batch)

| arm | variant | params | passes/step | source |
|---|---|---|---|---|
| A r=2, 1 layer | `Vanilla` | 204,373 | 1 | stored `runs/rank_mi` (0/8 SOLVED) |
| **L r=2 + loop x4** | `Looped` | **204,373** | 4 | NEW, 8 seeds |
| L4 r=2, 4 real layers | `Vanilla --n-layers 4` | 799,189 | 4 | NEW, 8 seeds (compute/params control) |
| C r=4, 1 layer | `Vanilla_r4mi` | 204,757 | 1 | stored `runs/rank_mi` (8/8) reference |

Gated: **L is bit-identical to A at every seed** -- same keys, same parameter count, max weight
difference 0.0; it IS A's weights with the block applied four times. Causal leak 0. So A vs L is
the cleanest contrast available: same parameters, same initialisation, same data, same recipe,
differing only in how many times the block runs. L4 adds 3.9x the parameters as well as the
compute, so it controls for "4x compute" but not at matched size.

Recipe as `runs/rank_mi`: batch 16, lr 1e-3, warmup + cosine, `--data-workers 3`, `--save-full-state`.
One in-batch **reproduction of A seed 0** must match the stored per-epoch losses exactly (the code
has gained new registrations since that batch).

## Readouts (budget-scoped: SOLVED = mean loss over the final 5% of epochs < 0.05)

Co-primaries, L vs A: SOLVED count (Fisher exact, two-sided) and T=1024 revisit accuracy (exact
permutation test with test-inversion 95% CI), `stats_core`. Secondary: L4 vs A and L vs L4 (same
tests, Holm over the three); T=512 / 2048; strata at T=1024; action code in angle space; and the
loop-count sweep at eval time (free: `n_loops` is a runtime argument -- rule 13).

## Branches, fixed now

- **H1 CONFIRMED** -- L vs A fires (either co-primary, p < 0.05) with L >= 4/8 SOLVED: rank 2's
  deficit is search, and iterations substitute for rank at IDENTICAL parameters. The rank
  recommendation becomes "spend iterations or rank", and the same account covers EM's recency
  deficit (`SEARCH_RESULTS.md`).
- **H1 REFUTED** -- L within MDE of A on both, and L <= 2/8: a search aid that works elsewhere does
  not recover rank 2 here. "Search deficit" then means something narrower than "any search aid
  helps", and the existence/stability result stands alone.
- **UNMEASURED** otherwise.
- **Compute reading** (secondary, read only after the primary): if L4 also fires, 4x compute is
  sufficient and the loop is not special; if L fires and L4 does not, weight sharing is doing
  something depth does not.

Void: the reproduction differs in any weight; any of the 17 runs missing; the md5 guard trips.

Scope: torus, T=1024, n_heads 2, d 128, one recipe, 900-epoch budget, one loop count (4).
Cost: L 7.4 s/epoch and L4 ~7 s/epoch solo (against A's 2.0), so ~2.5 h per run at 2 jobs/GPU;
17 runs over 4 slots ~ 10 h.
