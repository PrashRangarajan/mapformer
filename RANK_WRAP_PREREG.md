# Is rank D+1's 3D shortfall about dimension or about wrap-around? -- pre-registration (2026-10-01, before any run)

## Question
`RANK_ND_RESULTS.md`: per-head rank D+1 solves the torus 8/8 in 2D (grid 32) but 4/8 in 3D (grid 10), and every
failing arm fails on WRAP-ONLY revisits (the cell was seen before only at a different unwrapped position,
so the phase must be exactly periodic in N). The two cells differ in dimension AND wrap share (0.36 vs 0.70;
Amendment 1 there). This batch crosses them: dimension x wrap share, at rank D+1.

## Design (one batch, 8 seeds 0-7, rank D+1 only, the rank recipe at T=1024 as in RANK_ND)
| cell | D | grid | cells | wrap-only share (T=1024) | variant |
|---|---|---|---|---|---|
| 2L | 2 | 32 | 1024 | 0.361 | `Vanilla_r3ph` |
| 2H | 2 | 10 | 100 | 0.670 | `Vanilla_r3ph` |
| 3L | 3 | 18 | 5832 | 0.326 | `Vanilla_r4ph` |
| 3H | 3 | 10 | 1000 | 0.703 | `Vanilla_r4ph` |
Wrap shares from `docs/audits/2026-09-27/nd_floor_wrap.py`. Gates (`validate_nd --steps 1024 --n-traj 60`):
2:10 chance 0.515, best n-gram 0.510, revisit 0.902; 3:18 chance 0.499, n-gram 0.504, revisit 0.212 -- PASS;
2:32 and 3:10 as in RANK_ND. Cells 2L and 3H repeat RANK_ND's B2 and B3 with identical code and seeds: their
per-epoch losses must match `runs/rank_nd` exactly (a determinism check, not a replication).
Retrace floors at T=1024 (same script): 2L 0.750, 2H 0.591, 3L 0.820, 3H 0.653.

## Readouts and branches (SOLVED = final-5% loss < 0.05; accuracy = `eval_nd.py`, held-out map, T=1024)
A contrast FIRES if Fisher on SOLVED or exact permutation on accuracy has p < 0.05 in the stated direction.
- wrap effect within D: 2L vs 2H, 3L vs 3H (low wrap better).
- dimension effect within wrap level: 2L vs 3L, 2H vs 3H (2D better).
Branches, fixed now:
- **WRAP DRIVES IT** -- the wrap effect fires in both D, and the dimension effect fires at neither wrap level.
- **DIMENSION DRIVES IT** -- the dimension effect fires at both wrap levels, and the wrap effect fires in neither D.
- **BOTH** -- both effects fire in both comparisons.
- Anything else: reported as it falls.
Secondary (no verdict): held-out accuracy on wrap-only vs other revisits; own-training-map vs held-out
accuracy (memorisation: the 2H map has only 100 cells); T=2048; the 2L/3H determinism check.
Caveats fixed now: grid size changes cell count (100 to 5832), omega's init range (it follows the grid) and
revisit rate together with the wrap share; the design separates D from wrap share, not from those.
Void: any run missing; md5 guard trips; 2L or 3H losses differ from RANK_ND's (non-determinism).
Scope: rank D+1 only, T=1024, 900 epochs, n=8. Cost ~9 h (8 concurrent, data-bound).
