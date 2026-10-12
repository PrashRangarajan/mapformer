# Rank tests: the 1D ring and the escape test -- results (2026-10-11)

Pre-registration `RANK_TESTS_PREREG.md` + Amendment 1 (independent code audit, committed before the pilot was read) +
Amendment 2 (pilot, no change). Batch launched 16:00 at 9299470 (clean), done 21:55; md5 guard passed before eval and
analysis. Registered output `RANK_TESTS_ANALYSIS.txt`, `RANK_TESTS_VERDICTS.json`; declared secondaries
`docs/audits/2026-10-11/rank_tests_secondary_out.txt`; evals `RANK_TESTS_ESCAPE_{E,C,F}{,_RESCORED}.json`.

## Experiment 1, registered: RANK-D DIFFICULTY IN 1D BROKEN
On the 1D ring (32 cells, symmetric walk, T = 128, 300 epochs, 1 layer), per-head rank 1 -- rank equal to the space's
dimension -- solves **7/8**, against 8/8 at rank 2 (Fisher p 1.00). E2 (what the obstacle is) is NOT READ, because rank 1
does not fail.

| arm | per-head rank | SOLVED | acc T=128 / 512 | epochs to SOLVED (median) |
|---|---|---|---|---|
| R1 | 1 | 7/8 | 0.993 / 0.976 | 58 |
| R2 | 2 | 8/8 | 1.000 / 0.999 | 50 |
| R1DF (drift forbidden) | 1 | 8/8 | 1.000 / 1.000 | 24 |
| R1AO (observation steps 0) | 1 | 8/8 | 1.000 / 0.999 | 52 |
| R2S (shared rank 2: 2 latent dims, as R1) | 2 | 7/8 | 0.994 / 0.980 | 66 |
| floor (CANCEL) | | | 0.554 | |

**What it means.** The rank theory T1 predicted that a head whose rank equals the space's dimension rarely finds the
clean map, as on the 2D torus (rank 2: 0-2/8) and the 3D torus (rank 3: 1/8). In 1D it does: six rank-1 seeds reach a
loss near 0 by epoch ~50-75 with drift kappa ~1e-4 (clean), as fast as rank 2. So "rank = D is hard" is not a general
law. The 2D and 3D results stand as measured, but T1's account of them -- a rank-D head cannot place a clock off the map
-- does not carry to 1D. The tasks also differ (ring T = 128, batch 128, short gaps, 11% wrap-only revisits; torus
T = 1024), so the 1D result does not say which of the task or the dimension is responsible.

## Experiment 2, registered: X1 PARTIAL (largely unmeasured at this budget)
Lifting the 8 unsolved rank-2 runs to rank 4 (E) solves **3/8** in 600 epochs, against 1/8 for rank 2 continued (C;
Fisher p 0.57) and **2/8 for a fresh rank-4 step map on the same trained network (F)**. X2 and X3 are not read. The
registered guard did its job: F, the check that rank 4 can solve within the budget from these networks, solves only
2/8, so neither "rescue" nor "no rescue" is measurable at 600 epochs.

| arm | SOLVED | final-5% loss (per seed) | held-out acc T = 1024 (eval / re-scored) |
|---|---|---|---|
| E: rank-2 weights lifted to rank 4 (function-preserving) | 3/8 | 0.19 0.02 0.02 0.02 0.06 0.21 0.28 0.35 | 0.969 / 0.969 |
| C: rank 2 continued | 1/8 | 0.52 0.01 0.35 0.38 0.19 0.41 0.50 0.58 | 0.898 / 0.898 |
| F: trained network, fresh rank-4 step map | 2/8 | 0.12 0.02 0.16 0.16 0.17 0.11 0.01 0.06 | 0.975 / 0.975 |

## Secondaries (declared; no verdict)
- **Escape by source class.** Every E solve comes from a source that was still descending at 900 epochs (s1-s3); the
  four sources that had stalled stay unsolved in every arm (E, C, F: 0/4 each). Paired McNemar E vs C: 2 vs 0
  discordant, p 0.50.
- **The lifted models keep the clock; the fresh step maps do not.** At the end, every unsolved E run has a CLOCK head
  (5/5) although it uses the new directions (new / old W_out column norm 1.4-2.8); every F run, solved or not, is CLEAN
  (8/8) and still descending. With extra rank, the stalled rank-2 solution's drift persists; a fresh step map on the
  same network does not drift. Suggestive of a basin committed by the end of rank-2 training, but F did not converge
  either, so it is not a registered finding (X3 not read). "SOLVED iff CLEAN": E 8/8, C 8/8; F 2/8 (clean but slow).
- **Accuracy.** Both rank-4 arms are well above rank 2 continued (E 0.969, F 0.975, C 0.898); F is not below E, so the
  stalled rank-2 step map gives no visible head start (descriptive).
- **Ring.** R1's two slow seeds are the ones with a drifting head (kappa 0.04 and 0.06; one solved only at the end, one
  still descending at 0.16); the six fast ones are clean. Removing the action-common component (R1DF) roughly halves the
  epochs to solve (24 vs 58); removing only observation steps (R1AO) is about as fast as rank 2 (52 vs 50). r(final loss, acc) -0.990.
- Eval mode vs dropout-scale re-score: identical to 3 decimals in every arm (all near ceiling or the effect is absent).

## Caveats
- The ring differs from the torus in task and recipe, not only in D; the result breaks "rank D is hard" as a general
  rule, not the 2D / 3D measurements.
- 600 epochs was not enough for rank 4 on these trained networks (F 2/8): the escape question remains open.
- n = 8 per arm; the source set is the 8 RANK_MI rank-2 runs (4 stalled, 4 descending at 900 epochs).
