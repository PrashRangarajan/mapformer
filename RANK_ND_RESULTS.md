# Does the hard rank track the task's dimension? -- results (2026-10-01)

Pre-registration `RANK_ND_PREREG.md` (+ Amendment 1, committed before any result); runs `runs/rank_nd`
(32 runs, one batch, 8 seeds); eval `RANK_ND.md` / `.json`; registered output `RANK_ND_ANALYSIS.txt`
(`analyze_rank_nd.py`); declared secondaries `RANK_ND_SECONDARY.txt` / `.json`
(`analyze_rank_nd_secondary.py`); floors `docs/audits/2026-09-27/nd_floor_wrap.py`. Code-unchanged check at
read time (S5): every guarded file matches `runs/rank_nd/code_md5.txt`, and the analysis files are unchanged
since fcad5d5 (`docs/audits/2026-09-27/rank_nd_codecheck.txt`).

## Registered: no branch -- rank = D is hard in both dimensions; rank D + 1 rescues 2D fully, 3D only partly

N-D torus, trained and tested at T=1024, 900 epochs, per-head rank, every arm from the same r=2 base.

| cell | D | per-head rank | SOLVED | T=1024 acc | T=2048 | retrace floor (T=1024) |
|---|---|---|---|---|---|---|
| A2 | 2 (grid 32) | 2 (= D) | 1/8 | 0.769 +/- 0.121 | 0.689 | 0.750 |
| B2 | 2 (grid 32) | 3 (= D + 1) | **8/8** | **0.999** | 0.976 | 0.750 |
| A3 | 3 (grid 10) | 3 (= D) | 1/8 | 0.800 +/- 0.092 | 0.704 | 0.653 |
| B3 | 3 (grid 10) | 4 (= D + 1) | 4/8 | 0.847 +/- 0.176 | 0.810 | 0.653 |

- **Control (2D) holds strongly:** B2 - A2 SOLVED 8/8 vs 1/8 (Fisher p 0.0014), accuracy +0.230 (permutation
  p 0.0014). Rank = D fails and rank D + 1 solves in this environment as on the paper torus.
- **3D:** rank 3 (= D) fails like rank 2 in 2D (1/8). Rank 4 solves 4/8; B3 - A3 is +0.047 accuracy (p 0.50),
  4/8 vs 1/8 (Fisher p 0.28): **does not fire**. So RANK 3 IS ENOUGH is clearly ruled out (A3 1/8, not >= 6/8),
  and THRESHOLD TRACKS DIMENSION is not established because one extra dimension is not enough to rescue 3D
  within this budget. Read: **rank equal to the dimension is the hard case in both 2D and 3D**; rank D + 1 is
  sufficient in 2D (8/8) but only partly in 3D (4/8, unmeasured against rank D).

## Where the failures are: the wrap-only revisits (declared secondary S2)
Held-out accuracy at T=1024 split into WRAP-ONLY revisits (the cell was seen before only at a different
unwrapped position, so the phase must be exactly periodic in the torus size) and the rest:

| cell | wrap share | wrap-only | other revisits | own training map (S3) | gap own - held-out |
|---|---|---|---|---|---|
| A2 (D=2, r=2) | 0.37 | **0.530** | 0.907 | 0.937 | +0.169 |
| B2 (D=2, r=3) | 0.37 | 0.999 | 1.000 | 1.000 | -0.000 |
| A3 (D=3, r=3) | 0.71 | **0.729** | 0.974 | 0.917 | +0.118 |
| B3 (D=3, r=4) | 0.71 | **0.803** | 0.958 | 0.942 | +0.093 |

- Every arm that fails does so on the **wrap-only** revisits; the others are 0.91-0.97 across all arms. The
  rank-D difficulty is learning a phase code that is exactly periodic on the torus, consistent with the
  paper-torus rank line's wrap-only stratum (`RANK_MATCHED_RESULTS.md`: 0.557 vs 0.970 for r=2 vs r=4).
- The failing arms do better on their own training map (+0.09 to +0.17): they partly memorised the 1000-cell
  map instead of reading it from context. The solved arm (B2) has no gap. No SOLVED run is flagged (all SOLVED
  runs score >= 0.95 on the held-out map).
- 3D's lower B3 rate is in the regime with 71% wrap-only revisits; whether that, rather than dimension, is what
  makes rank 4 insufficient is not separated here (Amendment 1).

## Caveats
- The 2D and 3D rows differ in grid size, wrap share, omega's init range and initial step scale (Amendment 1);
  the cross-D reading is "in this setting", not a pure effect of dimension.
- Budget-scoped (900 epochs): many failing runs were STALLED or still DESCENDING.
- Accuracy levels are compared only within D; floors (retrace 0.750 / 0.653) are far above chance (0.52).
- n = 8; B3 vs A3 is UNMEASURED, not "no difference".
