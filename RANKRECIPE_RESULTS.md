# Does shared rank 2 still fail the T = 1024 torus at the best of three recipes? -- results (registered, `RANKRECIPE_PREREG.md` + Amendment 1)

Batch 2026-10-24 03:59 -> 21:10 on GPU 1 only, code `bed9fbf` (clean tree, md5 guard over 103 files, re-checked before every
launch), torch 2.10.0+cu128, 32 runs, fresh seeds 2600-2607: shared rank 2 (`Vanilla`) at three recipes and shared rank 4
(`Vanilla_r4mi`) at the default, torus 64 x 64, T = 1024, 900 epochs, cosine, the RANKWD recipe otherwise. Independent results-
blind check before launch (Amendment 1: 2 MAJOR, 8 MINOR, 4 NOTE, all folded in). Existence: a rank-2 solution exists and is held
(RANK_PROJ, RANKCLOCK), and the construction held a 60-epoch warm start at both new recipes. Pilot (seeds 2692-2693): lr 3e-3 not
UNSTABLE, so R1 stayed 3e-3. Analysis `analyze_rankrecipe.py` -> `RANKRECIPE_ANALYSIS.txt`, `RANKRECIPE_VERDICTS.json`; accuracy
`RANKRECIPE_<cell>.md` / `.json`.

**Reading (registered): RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH.** Rank 2 solves **0/8 at every recipe**; rank 4 solves
8/8 at the default. Rank 2 at its selected best recipe (lr 3e-3) against rank 4: SOLVED 0/8 vs 8/8, decision p 0.0002; accuracy
at T = 1024 0.9095 vs 0.9999 (d -0.090), decision p 0.0002. The search favours rank 2 (it picks rank 2's best of three while
rank 4 gets one recipe), so a cost that survives it is conservative. **Scope:** "the best of these three recipes" -- both
alternatives are higher-decay recipes (decay exponents 2.41 / 7.24 / 9.66), there is no lower-decay or combined lr + wd arm, and
the higher-lr arm sits at the grid edge.

## Gates and cells

G0 rank 4 8/8 (>= 6): pass. G1 rank 2 at the default 0/8 (<= 2): pass -- the rank-2 difficulty replicates (RANK_MI 0/8,
RANKCLOCK 1/8).

| cell | recipe | SOLVED | classes S / T / D | final-5% loss median (range) | acc T = 512 | acc T = 1024 | acc T = 2048 |
|---|---|---|---|---|---|---|---|
| r2_R0 | lr 1e-3, wd 0.05 (default) | 0/8 | 0 / 3 / 5 | 0.422 (0.101-0.849) | 0.888 | 0.860 | 0.803 |
| **r2_R1** | lr 3e-3, wd 0.05 | **0/8** | 0 / 3 / 5 | 0.383 (0.089-0.468) | 0.954 | **0.910** | 0.838 |
| r2_R2 | lr 1e-3, wd 0.2 | 0/8 | 0 / 6 / 2 | 0.531 (0.246-0.619) | 0.932 | 0.885 | 0.802 |
| r4_R0 | lr 1e-3, wd 0.05 (default) | **8/8** | 8 / 0 / 0 | -- | 1.000 | **1.000** | 0.997 |

Mean accuracy over 8 seeds (rank 4 at T = 1024: 1.000 +/- 0.000). No run was UNSTABLE or DIVERGED. Selection picked R1 (most
SOLVED is tied at 0, so the lowest mean final-5% loss, 0.370, decides); it is also rank 2's best recipe on accuracy, so the
choice does not depend on the tie-break.

## Primary (registered)

| | rank 2 best (r2_R1) | rank 4 (r4_R0) | test |
|---|---|---|---|
| SOLVED | 0/8 | 8/8 | naive Fisher 0.0002; selection-aware 0.0002; decision 0.0002 |
| acc T = 1024 | 0.9095 | 0.9999 | d -0.0904 (<= -0.05); naive permutation 0.0002; decision 0.0002 |

Both routes fire. The selection-aware p equals the naive p here (rule 3 / Amendment 1 finding 3: the naive p decides in practice).
r2_R1 vs rank 2's own default: 0/8 vs 0/8, Fisher p 1.0 -- no recipe changed the SOLVED count.

## Checks on the reading

- **Budget (rules 3, 4):** "within 900 epochs". 12 of the 24 rank-2 runs are DESCENDING (5 / 5 / 2 by cell); the higher-lr arm
  moved accuracy up (0.860 -> 0.910) and loss down without solving. A rank-2 recipe that solves later is outside this claim; the
  1800-epoch rank-2 result already on record (0/8 vs rank 4 7/8, `LOOP_RANK_E1800_P1_RESULTS.md`) points the same way.
- **Rule 2:** r(log10 final loss, acc) = -0.629 / -0.793 / -0.925 at T = 512 / 1024 / 2048 over 32 runs. The primary fires on
  SOLVED as well as accuracy.
- **Within batch:** one GPU, one code version; the rank-2 cells share every initial weight at a seed, and rank 4 shares all
  non-bottleneck weights.

## Descriptive: effective per-move angle

| cell | growth end / init | max angle per move (wrapped) |
|---|---|---|
| r2_R0 | 0.566 | 3.98 rad (2.78) |
| r2_R1 | 0.555 | 3.05 rad (2.90) |
| r2_R2 | 0.296 | 2.53 rad (2.53) |
| r4_R0 | 0.504 | 2.09 rad (2.09) |

All four cells carry decay and all keep their angles bounded (growth 0.3-0.6, unlike RANKWD's no-decay arm at ~9x), so the
rank-2 failure is not the angle blow-up RANKWD's correction described. Stronger decay (R2) shrinks the angles most and leaves the
most runs STALLED. A magnitude, not the function; moves only.

## What this changes

- **The rank claim survives a recipe search.** "Rank D+1 is needed" on the T = 1024 torus is not an artefact of the default
  recipe: rank 2 fails at the default, at 3x the learning rate and at 4x the weight decay (0/24), while rank 4 solves 8/8 at the
  default. The paper can state the rank result as robust to these three recipes, scoped as above.
- **Not established:** that no recipe rescues rank 2 -- the grid has no lower-decay or combined arm, and 12/24 rank-2 runs were
  still descending at 900 epochs.
