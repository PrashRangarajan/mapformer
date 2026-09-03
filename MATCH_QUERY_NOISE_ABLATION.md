# Context destruction on Match-Query, clean AND noisy

128^2, T_explore=512, T_query=256, chance 0.0625. Same checkpoints, same scored positions and
answers in every condition -- answers derive from the TRUE walk, which is
never altered.

| condition | control | explore-obs | query-actions |
|---|---|---|---|
| p0 · Vanilla | 0.493 | 0.050 | 0.086 |
| p0 · Looped | 0.824 | 0.061 | 0.079 |
| p010 · Vanilla | 0.302 | 0.069 | 0.142 |
| p010 · Looped | 0.366 | 0.079 | 0.167 |

## Verdict

**Clean Match-Query PASSES, decisively.** Both destructions take it to the chance
rate: explore-obs 0.050 / 0.061 and query-actions 0.086 / 0.079 against a chance of
0.0625, from controls of 0.493 and 0.824. The score depends on the explore
observations AND on the query path, which is what the task claims to measure. This
is the check hier-goal failed (0.912 -> 0.913).

**The NOISY variant at p=0.10 does NOT pass cleanly, and this run's automatic
verdict was wrong.** The rule was "fails if a destroyed condition keeps more than
half its control accuracy". query-actions keeps 0.142/0.302 = 47% and
0.167/0.366 = 46% -- under an ARBITRARY 50% line, so it printed PASSES. A threshold
that a result clears by three points is not a test. Read the numbers:

| p=0.10 | control | query-actions destroyed | chance | never-moved floor |
|---|---|---|---|---|
| Vanilla | 0.302 | **0.142** | 0.0625 | 0.104 |
| Looped | 0.366 | **0.167** | 0.0625 | 0.104 |

With the query path destroyed the model cannot know which cell it is being asked
about, yet it still scores 2.3-2.7x chance and above the never-moved baseline. The
explore phase leaks a LOCAL OBSERVATION PRIOR: having wandered a region, the model
can guess which observations are common there without localising at all. Roughly
**45% of the p=0.10 score is non-positional**.

**Consequence.** The right floor for the p=0.10 columns of MQ_NOISE_2X2.md and
MQ_NOISE_2X2_C2.md is ~0.15, not 0.0625. Above that floor the arms are Vanilla
~0.16 and Looped ~0.20, so the loop's +0.057 is a smaller share of real signal than
it looked, though the ORDERING is unchanged and the primary contrast (Level15 -
Vanilla, flat in drift) is a difference that largely cancels a common floor.

**The clean condition is unaffected** -- it collapses to chance, so the clean
columns stand as measured.

**Why this was missed until now.** The clean task passed this ablation once, and
the noisy variant was treated as inheriting that pass. It is a different task: drift
removes the positional route and leaves the distributional one relatively more
useful, which is exactly the substitution a destruction test exists to catch.
Gate every variant, not every family.

## Scope

n=4 seeds per cell, 2 arms, inference only. The never-moved floor (0.1042) is from
a 600-episode gate run; an earlier 120-episode estimate gave 0.1204 and was biased
high, which is worth remembering before quoting any small-sample floor.
