# Rank at matched initialisation -- results (2026-09-24)

Pre-registration `RANK_MI_PREREG.md`; runs `runs/rank_mi` (+ `runs/rank_mi_repro`); full output
`RANK_MI_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank_mi`). Every arm is built from our
r=2's initial weights at the same seed; only the bottleneck differs. Torus, trained and tested
at T=1024, 900 epochs, 8 seeds.

## Both registered questions fire

| arm | per-head content->angle rank | SOLVED within 900 ep | T=1024 accuracy |
|---|---|---|---|
| A our shared r=2 | 2 (2 latent dims, shared) | **0/8** | 0.894 +/- 0.070 |
| B the paper's per-head r=2 | 2 (4 latent dims, 2 per head) | **2/8** | 0.885 +/- 0.131 |
| C our shared r=4, matched init | 4 (4 latent dims, shared) | **8/8** | 0.998 +/- 0.005 |

| contrast | SOLVED (Fisher p) | accuracy diff (permutation p, 95% CI) |
|---|---|---|
| C - A | 8/8 vs 0/8 (**0.0002**) | +0.104 (**0.0005**, [+0.052, +0.155]) |
| C - B | 8/8 vs 2/8 (**0.0070**) | +0.112 (**0.0070**, [+0.021, +0.206]) |
| B - A | 2/8 vs 0/8 (0.47) | -0.009 (0.87, [-0.123, +0.100]) |

- **Q1: the r=4 advantage survives matched initialisation.** The stored r=4's 8/8 was not
  its different initial draws: built from r=2's own initial weights, r=4 again solves 8/8.
- **Q2: the paper's own per-head r=2 has the search problem too.** It solves 2/8, not
  detectably different from our shared r=2 (0/8), and detectably worse than r=4 (8/8).

**Reproduction check passed exactly again:** our r=2 seed 3 retrained in this batch matches
the stored run on all 900 epochs (max loss difference 0.0), so reusing the stored r=2 arm
and the pilot's per-head seeds is sound.

## What it means

B and C have the SAME number of latent dimensions (4); what differs is how many drive each
head's angle -- 2 in B (block-diagonal W_out) and in A, 4 in C. The arms line up with that
per-head rank, not with the total latent size: **the rank of each head's content-to-angle map
is what decides whether training finds the solution.** With the earlier results (a rank-2
solution exists, 0.9955, and is held under training, `RANK_PROJ_RESULTS.md`), the finding is:

> At rank 2 per head -- the paper's own design -- MapFormer usually fails to FIND the
> T=1024 torus solution within 900 epochs, though it can represent and hold it. At rank 4
> it finds it on every seed. "Use r=4" is advice about MapFormer's design, not about our
> reimplementation.

Scope and caveats:
- **Budget-scoped:** "within 900 epochs at T=1024" (Amendment 2 classes; runs still
  descending count as not solved). Nothing here says what the unsolved runs would do given
  more training.
- **Unseparated (registered):** C's W_out starts at a smaller scale (bound 0.5 vs 0.707 for A
  and B), so "rank 4 per head" and "smaller W_out init" move together in C (rule 15). A
  rank-2 arm initialised at W_out bound 0.5 would separate them.
- One task (torus navigation), one width (d=128, 2 heads), one recipe.
- Secondary, same direction: T=512 A 0.934 / B 0.910 / C 1.000; T=2048 0.819 / 0.832 / 0.973;
  long-gap revisits 0.671 / 0.784 / 0.995; wrap-only 0.557 / 0.644 / 0.977.
- Angle-space action code (`RANK_MI_GEOMETRY.md`): opposition A 1.13, B 0.93, C 0.053. It
  is not invariant to 2pi/omega wraps (rule 34): per-head seed 1 solved with opposition 0.78.
