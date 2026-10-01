# Does the hard rank track the task's dimension? -- pre-registration (2026-10-01, before any run)

## Question
On the 2D torus at T=1024 (`RANK_SEP_RESULTS.md`, `RANK3_RESULTS.md`), per-head rank 2 -- equal to the
task's 2 degrees of freedom -- is found on 0-2 of 8 seeds, rank 3 on 6/8, rank 4 on 8/8. Is "rank = the
number of dimensions" the hard case in general (then in 3D rank 3 fails and rank 4 is found), or is rank
3 simply enough (then rank 3 is found in 3D too)? Rank below the dimension cannot represent the walk at
all (`environment_nd.py` docstring: the step map has a kernel), so it is not tested.

## Task (`environment_nd.GridWorldND`; gate `validate_nd.py --configs 3:10,2:32 --steps 1024`, run 2026-10-01)
D-dimensional torus, GridWorld's directed walk, K=16, p_empty 0.5, revisit-scored. D=2: grid 32 (1024
cells); D=3: grid 10 (1000 cells). Gate at T=1024: chance 0.520 / 0.520, best action-stream n-gram 0.525 /
0.520, revisit rate 0.455 / 0.490 (D=2 / D=3) -- PASS. NOTE this is not the paper torus (grid 64) the 2D
rank results were measured on, hence the in-batch 2D control.

## Arms (one batch, 8 seeds 0-7, every arm built from our r=2 base at the seed like the RANK_SEP arms)
| cell | D | per-head rank | variant |
|---|---|---|---|
| A2 | 2 | 2 (= D) | `Vanilla_r2ph` |
| B2 | 2 | 3 (= D + 1) | `Vanilla_r3ph` |
| **A3** | 3 | 3 (= D) | `Vanilla_r3ph` |
| **B3** | 3 | 4 (= D + 1) | `Vanilla_r4ph` |
Recipe = the rank line's at T=1024: batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, 1 layer,
2 heads, d 128, `--data-workers 3`, `--save-full-state`, `--env nd --n-dims D --grid-size N`. 32 runs.

## Readouts and branches (SOLVED = final-5% loss < 0.05; accuracy = `eval_nd.py`, held-out env seed
## 10000, T=1024 registered, T=2048 secondary; tests: Fisher on SOLVED, exact permutation on accuracy)
Control (D=2): B2 vs A2. Primary (D=3): B3 vs A3. A contrast FIRES if either test p < 0.05 (rank+1 higher).
- **CONTROL FAILED** -- A2 SOLVED >= 6/8: rank = D is not hard in this environment even in 2D; the D=3 row
  cannot be read as a shift and is reported as it falls.
- **THRESHOLD TRACKS DIMENSION** -- control holds (A2 <= 2/8), A3 <= 2/8, and B3 vs A3 fires.
- **RANK 3 IS ENOUGH** -- control holds, A3 >= 6/8, and B3 vs A3 does not fire.
- Anything else: reported as it falls.
Secondary: T=2048; final-loss regimes per cell; the per-run action geometry is not registered.
Void: any run missing; md5 guard trips.
Scope: torus with D in {2, 3}, ~1000 cells, T=1024, n_heads 2, d 128, one recipe, 900 epochs, n=8.
Cost: ~3.5 s/epoch alone (data-bound), 8 concurrent on 32 cores; ~5-6 h.

---

## Amendment 1 (2026-10-01, after an independent CPU audit, BEFORE any result of the batch was read)
No bug changes the registered computation. Dates: this pre-registration was committed (fcad5d5) at
2026-09-30 22:59, the same second the driver started; "2026-10-01" in its header is wrong. Measured cost
is ~9 h (8.3-9.0 s/epoch with 8 concurrent jobs), not 5-6 h. Qualifications, fixed now:
1. **The two rows differ in more than D.** Wrap-only revisits (seen before only at a different unwrapped
   position, so the phase must be exactly periodic in N) are 0.361 of revisits at D=2 N=32 and **0.703**
   at D=3 N=10 (the paper torus, N=64, where the 2D rank results were measured: 0.080). Omega's init range
   follows N (lowest omega 0.196 vs 0.628), and the initial angle-increment std ratio rank D+1 / rank D is
   1.00 (D=2) vs 0.90 (D=3). A THRESHOLD TRACKS DIMENSION verdict will therefore be read as "rank = D is hard
   at D=2 and D=3 in this setting", not as a pure effect of dimension; RANK 3 IS ENOUGH likewise.
2. **Floors.** The gate's chance / n-gram (0.52) understates what a non-path strategy reaches: a retrace
   predictor (copy o[t-2j] while a run reverses the previous one) scores 0.750 (D=2 N=32) and 0.653 (D=3
   N=10) at T=1024 (`docs/audits/2026-09-27/nd_floor_wrap.py`). Accuracy levels are compared only within D.
3. **Memorisation.** The training map has 1000 cells; SOLVED (training loss) could include memorising it.
Declared secondaries (`analyze_rank_nd_secondary.py`, run after the batch, no verdict):
- S1 the floors above beside every accuracy cell;
- S2 held-out accuracy on wrap-only vs other revisits, per arm;
- S3 accuracy on each run's own training map beside the held-out map; SOLVED runs with held-out accuracy
  below 0.95 flagged;
- S4 the 2D control's B2 - A2 contrast reported beside the verdict (the registered branch only requires
  A2 <= 2/8, not B2 > A2);
- S5 at read time: md5 of every guarded file against `runs/rank_nd/code_md5.txt`, and
  `git diff fcad5d5 -- stats_core.py analyze_rank_nd.py eval_nd.py environment_nd.py` (empty expected),
  recorded in the results file (the guard runs only at driver start).
The FIRES rule is a union of two two-sided tests with a direction condition: at most ~0.05 one-sided per
contrast; at n=8 Fisher fires only at 0 vs >= 5, 1 vs >= 6, 2 vs >= 7.
