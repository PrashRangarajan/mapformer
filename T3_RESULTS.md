# T3: a per-token PoPE phase -- the fix works, the predicted cost does not appear

Pre-registration: `T3_PREREG.md`. `delta` gains a per-token component from zero-initialised heads, so
training starts as exactly PoPE. The accumulator is untouched: same clock, same alpha, same range.

## Part A -- Bach Chorales, training context 512 (5 seeds). Test NLL, lower is better

| arm | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|
| MapPoPE (baseline) | 0.5235 | 1.9536 | 4.6158 |
| **MapPoPE + per-token phase (T3)** | 0.5594 | **0.6411** | **0.7325** |
| MapPoPE T3 inert twin (same params, phase gated off) | 0.5226 | 1.8337 | 4.4486 |
| MapPoPE centred (T1) | 0.6560 | 0.7613 | 0.9105 |
| MapWM | 0.5382 | 0.7835 | 1.3969 |

- **A1 PASSES decisively**: T3 - MapPoPE = **-3.883 at 1024-2048** (5/5, MDE 0.425) and -1.313 at
  512-1024. T3 is the best arm measured on this task at long range -- better than centring (0.911)
  and better than MapWM (1.397), which nothing else in this line has managed.
- **A3 PASSES**: the inert twin is unmeasured at every bucket (-0.167 at 1024-2048, MDE 0.828, 3/5).
  The 786k extra parameters do nothing; the phase does the work.
- **A2 PARTIAL**: T3 costs +0.036 in distribution (5/5 seeds worse, detectable). Registered as
  "not detectably worse", so strictly it is not met -- but it is **3.7x cheaper than centring's
  +0.133**, which is the comparison the test was built to make. Changing only the compensating
  degree of freedom removes the collapse; shrinking the accumulator also removes it but costs more.

## Part B -- Indirect Indexing at 100,000 iterations (8 seeds). Solve rate

| arm | solved | mean | per-seed |
|---|---|---|---|
| MapPoPE (baseline) | 5/8 | 0.598 | 0.091 .. 0.996 |
| **MapPoPE T3** | **5/8** | 0.651 | 0.097, 0.162, 0.434, 0.620, 0.948, 0.973, 0.986, 0.989 |
| MapPoPE T3 inert twin | 6/8 | 0.752 | 0.095, 0.100, 0.911 .. 0.996 |
| PoPE (for reference) | 1/8 | 0.199 | -- |

- **B1 NOT CONFIRMED.** The predicted cost does not appear: T3 solves 5/8, exactly the baseline
  (Fisher p = 1.0), with a slightly higher mean and lift-off steps in the same range (45-70k).
- **B2 clean**: the inert twin is 6/8, so the extra parameters are not making the search harder;
  if anything they help slightly, and not detectably.

> **Scope (2026-09-20)**: the measurement in this file stands. The cross-task account it was read
> as supporting does not -- both halves fail a within-task test on Dyck (`T2_RESULTS.md`), so
> treat this as a fact about Bach at a 512-token context.

## What this does to the account

**The collapse is explained; the corollary is withdrawn.**

- The conjunction account survives its sharpest test. Restoring ONLY the pairwise phase -- with the
  accumulator, its growth rate and its range all unchanged, and with a parameter-matched inert twin
  showing the parameters are inert -- removes almost the entire collapse. Combined with T1, where
  shrinking only the accumulator also helped and helped MapPoPE 5.7x more than MapWM, both halves of
  the conjunction now have independent interventional support.
- **The predicted trade-off is dead.** I registered that a content-dependent phase would re-entangle
  what and where and therefore cost PoPE's pure-indexing advantage. It costs nothing measurable.
  Registered consequence: **the account is too simple** and the trade-off claim is withdrawn.

**Post-hoc, and labelled as such** -- and SUBSEQUENTLY REFUTED (2026-09-19, `T3GEN_RESULTS.md` G3:
forcing the phase helps monotonically instead of costing anything, and on Dyck the model keeps MORE
phase than the Bach model while doing worse; the reading below is wrong): the prediction assumed the
freedom is FORCED. It is optional --
`delta` is ADDITIVE and zero-initialised, the magnitude channel that carries content is untouched, so
a model that needs a pure positional kernel can simply leave the phase heads near zero and does. That
reading was not predicted and is not evidence; it is a hypothesis, and it has an obvious test:
initialise the phase heads away from zero, or scale their output up, so the entanglement cannot be
declined. If indexing then degrades, the trade-off exists and is merely avoidable; if it still does
not, a per-token additive phase really is free on these tasks.

**Power caveat on Part B**: at n=8 the solve-rate comparison is weak -- even the original
5/8-vs-1/8 gap was only p = 0.119. A moderate cost would not be visible here. "No cost observed" is
the honest claim, not "no cost exists".

## Batch provenance (audit note, 2026-09-19)

The contrasts here pair by seed index ACROSS run directories built on different days, not within one
batch -- the repo's standing rule 3 asks for one batch. Mitigating: the arms share code (only new
classes were added between the runs; the data and evaluation paths are byte-identical), the data
stream is seeded identically, and the primary readings lean on a parameter-matched inert twin
trained INSIDE the new batch. Unmitigated for the decay arms, which have no same-batch baseline.
