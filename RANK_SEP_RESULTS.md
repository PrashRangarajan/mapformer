# Rank separation -- results (2026-09-25)

Pre-registration `RANK_SEP_PREREG.md`; runs `runs/rank_sep` (+ `runs/rank_sep_repro`); full output
`RANK_SEP_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank_sep`). All five arms are built from our
r=2's initial weights at the same seed and differ only in the bottleneck. Torus, trained and tested
at T=1024, 900 epochs, 8 seeds. SOLVED = final-5% training loss < 0.05.

## The split is entirely by PER-HEAD rank

| arm | variant | latent dims | each head's angle reads | SOLVED | T=1024 acc |
|---|---|---|---|---|---|
| A our shared r=2 | `Vanilla` | 2 (shared) | 2 | 0/8 | 0.894 |
| B per-head r=2 | `Vanilla_r2ph` | 4 (2 per head) | 2 | 2/8 | 0.885 |
| C_bd block-diagonal r=4 | `Vanilla_r4mibd` | 4 (shared, off-blocks frozen at 0) | 2 | 2/8 | 0.948 |
| C shared r=4 | `Vanilla_r4mi` | 4 (shared) | **4** | **8/8** | 0.998 |
| D per-head r=4 | `Vanilla_r4ph` | 8 (4 per head) | **4** | **8/8** | 0.999 |

Every arm whose per-head map has rank 2 solves 0-2 of 8; every arm at rank 4 solves 8 of 8.
Total latent size does not track it (B and C both have 4 dims and land on opposite sides).

## The registered contrasts settle the three candidates

| factor | contrast | SOLVED | Fisher p | accuracy | perm p | Holm | verdict |
|---|---|---|---|---|---|---|---|
| per-head rank (at matched sharing: both block-diagonal) | D - C_bd | 8/8 vs 2/8 | **0.0070** | +0.051 | **0.0070** | 0.028 | **FIRES** |
| cross-head reading (C's off-blocks removed) | C - C_bd | 8/8 vs 2/8 | **0.0070** | +0.049 | **0.0070** | 0.028 | FIRES, but it also drops per-head rank 4 -> 2, so it is not a separate factor |
| sharing (per-head latents vs one shared latent, both rank 4) | D - C | 8/8 vs 8/8 | 1.00 | +0.001 | 0.59 | 0.59 | UNMEASURED |
| W_out per-entry scale (both per-head rank 2) | C_bd - B | 2/8 vs 2/8 | 1.00 | +0.063 | 0.24 | 0.49 | UNMEASURED |

**Reproduction passed exactly:** C seed 0 retrained in this batch matches the stored `runs/rank_mi`
run's per-epoch losses on all 900 epochs (max diff 0.0).

## What this establishes

> **It is the rank of each head's content-to-angle map.** At rank 2 per head, MapFormer finds the
> T=1024 torus solution on 0-2 of 8 seeds; at rank 4 per head, on 8 of 8. Whether the heads share one
> latent or own separate ones makes no difference (D - C unmeasured), and `W_out`'s per-entry scale
> does not either (C_bd - B unmeasured). With `RANK_PROJ_RESULTS.md` (a rank-2 solution exists,
> 0.9955, and is held under training) this is a SEARCH deficit, not a capacity limit.

This **supersedes the "unseparated" caveat** in `RANK_MI_RESULTS.md`: the three candidates that
batch confounded are now separated, and the one that survives is per-head rank. The D - C_bd contrast
carries it, because those two arms are both block-diagonal and differ only in rank per head.

**The initial-angle-scale worry is answered.** Zeroing C's off-blocks halves C_bd's initial angle
increment (std 0.208 against ~0.33 for the others), so C_bd could in principle have failed for that
reason. It did not: A (0.335) and B (0.328) have normal initial scale, the same per-head rank 2, and
fail the same way. Low initial scale is not necessary for failure.

Scope: torus, T=1024, n_heads=2, d=128, 900-epoch budget, one recipe. Untested: rank 3 per head,
more heads, whether a rank-2 arm can be made to find the solution by another route (init, schedule,
curriculum).
