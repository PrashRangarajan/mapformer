# H1 part 1: is rank 2's failure a budget effect? -- results (2026-09-29)

Pre-registration `LOOP_RANK_E1800_PREREG.md` (Amendment 1: part 1 = A and C only; only branch
RANK 2 WAS BUDGET is read). Runs `runs/loop_rank_e1800` (A `Vanilla` r=2, C `Vanilla_r4mi` r=4, 8 seeds,
1800 epochs, from scratch, the rank recipe at T=1024); output `LOOP_RANK_E1800_P1.md` / `.json`,
`LOOP_RANK_E1800_P1_ANALYSIS.txt` (`analyze_loop_rank_e1800_budget.py`).

## Registered (branch 1): NOT a budget effect at 1800 epochs

| arm | SOLVED at 900 ep (`runs/rank_mi`) | **SOLVED at 1800 ep** | T=1024 acc 900 -> 1800 |
|---|---|---|---|
| A, r=2 | 0/8 | **0/8** | 0.894 -> 0.908 |
| C, r=4 | 8/8 | **7/8** | 0.998 -> 0.994 |

C - A at 1800: SOLVED 7/8 vs 0/8 (Fisher p 0.0014); accuracy +0.086 (permutation p 0.0003).
Doubling the budget does not rescue rank 2: final losses 0.14-0.57, none below 0.05. The citable rank
split survives at twice the budget.

## Caveats
- Still budget-scoped: 5 of the 8 rank-2 runs are DESCENDING at 1800 epochs (3 STALLED), so "never"
  is not shown; what is shown is that 2x the budget moves accuracy by +0.014 and solves none.
- One rank-4 run (s5) STALLED at 0.180 under the longer schedule, where all 8 solved at 900; a longer
  cosine schedule is not uniformly better.
- The registered H1 primary (loop vs plain rank 2) is NOT read; its arms are deferred (part 2).
