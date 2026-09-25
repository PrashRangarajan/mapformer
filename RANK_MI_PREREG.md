# Rank at matched initialisation -- pre-registration (2026-09-24, before any new run)

## Why

At T=1024 (900 epochs) our shared r=4 solved 8/8 and our shared r=2 0/8. But the stored
r=4 (`Vanilla_r4`) draws its wider W_in inside the base constructor, which shifts EVERY
later random draw: at the same seed it shares no initial weight with r=2 (max diff 0.18,
checked). The per-head pilot (`RANK_PERHEAD_PILOT_RESULTS.md`) tracked our r=2 seed for seed,
which may only reflect shared initial weights. This batch removes the confound.

## Arms -- every one built from our r=2's base at the same seed; only the bottleneck differs

| arm | variant | latent dims | bottleneck params | source |
|---|---|---|---|---|
| A our shared r=2 | `Vanilla` | 2 | 384 | stored `runs/rank_matched_e900` s0-7 (reused) |
| B the paper's per-head r=2 | `Vanilla_r2ph` | 4 (block-diagonal W_out) | 640 | pilot s0-1 reused, s2-7 new |
| C our shared r=4, matched init | `Vanilla_r4mi` | 4 (full W_out) | 768 | s0-7 new |

Gated: B and C share every non-bottleneck initial weight with A at each seed (max diff 0.0);
params 204,373 / 204,629 / 204,757; causal leak 0; C's W_out at nn.Linear's default (bound
0.5, as the stored r=4). Recipe of `runs/rank_matched_e900`: T=1024, batch 16, 900 epochs,
lr 1e-3, warmup + cosine, `--data-workers 3`, fresh start. Reuse is licensed because the
code is verified bit-identical to the code that produced the stored runs (short-run and
11-config checks, commits 64a463e..da0838f); **one full reproduction run**, A seed 3, is
retrained in this batch and must match the stored per-epoch losses exactly. B's seeds 0-1
are known in advance (1 solved, 1 stalled); this is stated, not hidden.

## The question, scoped to the budget

**Which bottleneck lets training find the solution WITHIN 900 epochs at T=1024?** A run is
SOLVED if its mean loss over the final 5% of epochs is < 0.05 (Amendment 2). Runs that are
still descending count as not solved within the budget; no "unreadable" rule applies, and no
sentence may claim what an unsolved run would do with more training.

## Readouts

Per arm: SOLVED count (co-primary, Fisher exact, two-sided) and T=1024 revisit accuracy
(co-primary, exact two-sample permutation test and test-inversion 95% CI, `stats_core`).
Pairwise: C vs A, B vs A, C vs B. Secondary: T=512 / 2048, strata at T=1024, action code in
angle space (`probe_action_geometry --space delta`), and C against the stored r=4 (does the
stored 8/8 hold at matched init?).

## Branches (alpha 0.05 on either co-primary test)

**Q1 -- the init confound (C vs A).**
- C > A significant: the r=4 advantage survives matched initialisation.
- Neither test significant and C solves <= 4/8: the stored r=4's 8/8 depended on its initial
  draws; the rank recommendation is withdrawn.
- Otherwise: unmeasured.

**Q2 -- the paper's own design (B against A and C).**
- C > B significant: the paper's per-head r=2 has the search problem too; a full 4-d latent
  shared across heads is what helps, and "use r=4" is advice about MapFormer's design.
- B > A significant and B vs C not: what matters is the latent dimension count (4 vs 2), not
  sharing across heads; the paper's per-head design is fine and our shared r=2 is the problem.
- Otherwise: unmeasured.

Not separated by this design: C's W_out initial scale (bound 0.5) differs from A's and B's
(0.707), so "4 dims" and "smaller W_out init" move together in C (rule 15).
