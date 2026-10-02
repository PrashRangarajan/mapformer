# Is rank D+1's 3D shortfall about dimension or about wrap-around? -- results (2026-10-02)

Pre-registration `RANK_WRAP_PREREG.md` (+ Amendment 1, committed before any result); runs `runs/rank_wrap`
(32 runs, one batch, 8 seeds); eval `runs/rank_wrap/N*/EVAL_D*.json`; registered output
`RANK_WRAP_ANALYSIS.txt` (`analyze_rank_wrap.py`), `RANK_WRAP.json`; amended reading `RANK_WRAP_AMENDED.txt`
(`analyze_rank_wrap_secondary.py`). Determinism: 2L and 3H reproduce RANK_ND's B2 and B3 per-epoch losses
exactly on 16/16 seeds (max diff 0.0).

## Registered: WRAP DRIVES IT (amended reading agrees) -- but only the 3D half is a clean measurement

| cell | D | grid (cells) | wrap-only share | SOLVED | held-out acc | wrap-only / other | own training map |
|---|---|---|---|---|---|---|---|
| 2L | 2 | 32 (1024) | 0.37 | 8/8 | 0.999 | 0.999 / 1.000 | 0.999 |
| 2H | 2 | 10 (100) | 0.68 | 1/8 | **0.273** | 0.243 / 0.334 | **0.986** |
| 3L | 3 | 18 (5832) | 0.35 | **8/8** | **1.000** | 1.000 / 1.000 | 1.000 |
| 3H | 3 | 10 (1000) | 0.71 | 4/8 | 0.847 | 0.804 / 0.958 | 0.942 |

Contrasts (Fisher on SOLVED; exact permutation on accuracy):
- wrap, 3D: 3L over 3H 8/8 vs 4/8 (p 0.077), accuracy +0.153 (p 0.0085), within type: wrap-only +0.196 (p 0.044),
  other +0.042 (p 0.026) -- FIRES.
- wrap, 2D: 2L over 2H 8/8 vs 1/8 (p 0.0014), +0.726 (p 0.0002) -- FIRES, but see below.
- dimension at low wrap: 2L vs 3L both 8/8, -0.001 (p 0.51) -- does not fire (both at ceiling).
- dimension at high wrap: 2H vs 3H -0.574 (p 0.0002) -- REVERSED (3D better), because of 2H below.

## What it shows
- **Rank D+1's 3D shortfall was not about dimension.** On a 3D torus with few wrap-around revisits (grid 18),
  rank 4 solves 8/8 at accuracy 1.000, exactly like rank 3 in 2D. The 4/8 of `RANK_ND_RESULTS.md` came from the
  small, wrap-heavy 10-per-side grid. Within 3D the high-wrap cell is worse on wrap-only revisits (+0.196) and
  slightly on the others (+0.042). Caveat: grid 18 vs 10 also changes the cell count (5832 vs 1000), the revisit
  rate (0.21 vs 0.49) and omega's init range, so "wrap share" here means "the small-grid regime".
- **The 2D high-wrap cell did not test wrap-around: it memorised its map.** With 100 cells and 90% revisits, the
  model learned the training map (0.986 on it) and scores 0.273 on an unseen map -- below always-blank (0.46)
  and on ordinary revisits too (0.334), not only on wrap-only ones. Its contrast with 2L is a memorisation
  effect, not a wrap effect, and the registered "WRAP DRIVES IT" rests on it for its 2D half. Only one of its
  runs reached SOLVED (training loss), flagged SOLVED-but-held-out < 0.95.
- Combined with `RANK_ND_RESULTS.md`: per-head rank = D is hard in 2D and 3D; rank D + 1 suffices in both when
  the torus is large enough that wrap-around revisits are a minority. On small wrap-heavy tori, rank D+1 is
  only partly enough (3D) or the model memorises instead (2D, 100 cells).

## Caveats
- The registered verdict's 2D half is invalid as a wrap test (memorisation); the clean evidence is the 3D pair.
- Grid size co-varies with wrap share, cell count, revisit rate and omega's init range (pre-registered caveat).
- n = 8; 900 epochs; 3H's failing seeds were STALLED or DESCENDING.
