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

> **CORRECTED 2026-09-25 after review.** An earlier version said "the rank of each head's
> content-to-angle map decides it". The design does not isolate that; the text below does not
> claim it.

B and C have the same total latent size (4) and, at initialisation, IDENTICAL `W_in`; they
differ only in `W_out`, and in three ways at once:
1. the rank of each head's content-to-angle map (2 in B, 4 in C);
2. whether the heads read a SHARED latent (at 2 heads and 4 dims, a head can have rank 4 only
   if both heads read the same latent, so 1 and 2 are perfectly confounded);
3. `W_out`'s per-entry scale (bound 0.707 vs 0.5) and the number of `W_out` coordinates feeding
   each angle (2 vs 4) -- under Adam, C's angle map can move up to ~2x further per step.

The INITIAL scale of the angle increments is matched (mean std over 8 seeds 0.335 / 0.343 /
0.344 for A / B / C: nn.Linear's default preserves fan-in variance), so "a smaller initial angle
map" is excluded; A and C start at the same scale and end with opposite outcomes.

What the batch supports:

> On the torus at T=1024, within 900 epochs and at matched initialisation, shared r=4 finds the
> solution on 8/8 seeds; our shared r=2 on 0/8; a per-head r=2 on 2/8. r=4's advantage is not
> its initial draws. Against the per-head r=2 it holds at the same total latent size and
> identical W_in; WHICH of per-head rank, cross-head sharing and W_out scale is responsible is
> unseparated. B - A is UNMEASURED (Fisher p 0.47, accuracy CI [-0.123, +0.100]).

"The paper's design": App. A.7 gives a per-head `W_in` but never states `W_out`'s shape. B is
the literal reading; if `W_out` is one full matrix over all heads' latents, the paper's r=2 at
2 heads IS arm C (8/8). Existence and stability (`RANK_PROJ_RESULTS.md`) were measured for our
SHARED r=2; that the per-head r=2 can also hold the solution is inferred.

The separating experiment (reviewer's design): **C_bd** -- C's own initial weights with the
off-diagonal blocks of `W_out` zeroed and kept at zero (a per-head r=2 at scale 0.5, `W_in`
identical to B and C): C vs C_bd isolates cross-head reading; C_bd vs B isolates `W_out` scale.
**D** -- per-head r=4 (8 latent dims, block-diagonal): D vs C isolates sharing, D vs C_bd isolates
per-head rank. 16 runs. (The earlier-proposed "rank-2 at W_out bound 0.5" would halve the initial
angle-map variance and add a confound; withdrawn.)

Scope and caveats:
- **Budget-scoped:** "within 900 epochs at T=1024" (Amendment 2 classes; runs still
  descending count as not solved). Nothing here says what the unsolved runs would do given
  more training.
- **Unseparated:** per-head rank, cross-head sharing and W_out's per-entry scale (above).
- One task (torus navigation), one length (trained and tested at T=1024), n_heads=2 only (per-head
  and shared designs differ only with several heads), one width (d=128), one recipe.
- Secondary, same direction: T=512 A 0.934 / B 0.910 / C 1.000; T=2048 0.819 / 0.832 / 0.973;
  long-gap revisits 0.671 / 0.784 / 0.995; wrap-only 0.557 / 0.644 / 0.977.
- Angle-space action code (`RANK_MI_GEOMETRY.md`): opposition A 1.13, B 0.93, C 0.053. It
  is not invariant to 2pi/omega wraps (rule 34): per-head seed 1 solved with opposition 0.78.
