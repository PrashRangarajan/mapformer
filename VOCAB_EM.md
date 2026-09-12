# The vocab sweep with a healthy EM arm: the capacity claim does NOT reproduce

5 arms x 3 vocabularies x 8 seeds, ONE batch, cosine / lr 1e-3 (not the published
recipe -- see P3). Torus, held-out map. Pre-registration: `VOCAB_EM_PREREG.md`.
Reading order fixed in advance: shape before mean.

## The shape, T=128 (the paper's regime -- position sharp, no drift)

| n_obs | arm | min | gap (worst->2nd) | sd | mean |
|---|---|---|---|---|---|
| **256** | `Vanilla` (WM r2) | 0.990 | 0.004 | 0.003 | 0.997 |
| | `Vanilla_r4` (WM r4) | **0.518** | **0.482** | 0.171 | 0.940 |
| | `VanillaEM` | 0.796 | 0.031 | 0.036 | 0.849 |
| | `VanillaEM_P0` | **0.504** | **0.225** | 0.171 | 0.849 |
| | **`VanillaEM_P0_r4`** | **0.995** | **0.005** | **0.002** | **0.999** |
| 64 | `VanillaEM_P0_r4` | 1.000 | 0.000 | 0.000 | **1.000** |
| 16 | `VanillaEM_P0_r4` | 1.000 | 0.000 | 0.000 | **1.000** |

**`VanillaEM_P0_r4` is 1.000 / 1.000 / 0.999 across all three vocabularies -- the
only arm that neither degrades nor collapses anywhere.**

## P1 REFUTED: the capacity win is RELIABILITY, not capacity

`EM_P0_r4 - Vanilla_r4` does grow across vocabulary -- `-0.000 / +0.000 / +0.060` --
which is the predicted profile. But it does not survive the check that matters:

| | full mean | drop each arm's worst seed |
|---|---|---|
| n_obs=64 | +0.0004 | **+0.0001** |
| n_obs=256 | **+0.0597** | **+0.0000** |

**The entire +0.060 is one collapsed `Vanilla_r4` seed.** Trimmed symmetrically, the
two arms are identical to four decimal places (1.0000 vs 0.9999). Every contrast is
also unmeasured at n=8 (MDE 0.169 at n_obs=256, inflated by exactly that collapse).

So on the one test aimed at the paper's quadratic-capacity claim, **a healthy EM arm
does not out-recall a healthy WM arm at any vocabulary.** What it does is never fail.

## P3: the collapse persists, and my init attribution was wrong HERE

`VanillaEM_P0` at n_obs=256, T=512, per-seed:
`0.491 0.654 0.755 0.812 0.875 0.949 0.955 0.957`

The published old-recipe run had `0.910 / 0.502 / 0.906`. **The low seed survives the
better recipe**, so that collapse was not the recipe -- but it was also not fixed by
shared `p_0`, which is what I attributed it to.

## The reversal: which fix binds is task-dependent

Gap at n_obs=256 (T=128), against the same statistic on MiniGrid:

| arm | torus n_obs=256 | MiniGrid |
|---|---|---|
| `VanillaEM` (neither fix) | 0.031 | 0.137 |
| `VanillaEM_P0` (origin only) | **0.225** | **0.016** |
| `Vanilla_r4` / `VanillaEM_r4` (rank only) | 0.482 (WM) | 0.098 |
| `VanillaEM_P0_r4` (both) | **0.005** | **0.002** |

**On MiniGrid the origin fix removed the collapse and rank alone did not. Here it is
the reverse.** Only the combination works on both. So neither fix is *the* fix --
they address different failures, and which one binds is a property of the task.

And a genuinely unexpected one: **`Vanilla_r4`, a WM arm, is the worst collapser
here** (min 0.518, gap 0.482) while EM with both fixes is the most stable thing in the
batch. The fragility I had been attributing to EM's AND-gate is not exclusive to EM.

## What survives

- **`VanillaEM_P0_r4` is the arm to use.** Best mean at every vocabulary, and the only
  one with no collapse anywhere -- on this task and on MiniGrid.
- **The paper's capacity claim has no support here.** Not refuted in general: this is
  one task at l=128 on a 64x64 torus, and the paper's scaling is at l=16 with
  vocabularies to 10,000. But the test aimed at it, with the pathology removed, shows
  parity rather than an advantage.
- ~~**"EM is worse" remains dead.**~~ **REFUTED THE NEXT DAY** (`RECENCY_EM_RESULTS.md`): on
  recency with k varying per query, single-`p0` EM - WM = **-0.375** (0/8) on the FIXED arms.
  Original text: Every prior EM deficit traced to an init pathology,
  and with both fixes EM is equal-or-better everywhere measured.

## Caveats

All contrasts unmeasured at n=8. `n_obs=16` and `64` are at ceiling for the r=4 arms
(1.000), so only n_obs=256 carries signal -- and there the difference is one seed.
Gaps and sds are descriptive, not tests. Recipe differs from the published sweep by
design, so these numbers are not comparable to `VOCAB_SWEEP_MULTISEED.md`.
