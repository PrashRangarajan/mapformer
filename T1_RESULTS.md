# READING -- the account survives its falsifier, with three caveats

**P1 and P2: the predicted asymmetry fires.** Centring the increment takes MapPoPE's far bucket from
4.616 to **0.911** (-3.705, 5/5 seeds, detectable) and MapWM's from 1.397 to 0.741 (-0.656, 5/5).
Both improve, but MapPoPE's gain is **5.7x larger**, which is the registered discriminator: the
failure condition was "if the two move by similar amounts the account is not supported". They do not.
The catastrophe was the accumulator leaving its calibrated range, and removing most of that removes
most of the catastrophe.

**P4: the combination finally works out of distribution** -- centred MapPoPE (0.911) beats UNCENTRED
MapWM (1.397), the first time in this line that PoPE plus path integration is ahead of path
integration alone at long range.

## Three caveats, none of which the result survives without

1. **M1 is only partially met.** alpha fell from 1.003 to 0.891 (MapPoPE) and 0.826 (MapWM), not to
   the ~0.5 of a true random walk. What centring actually bought was SCALE, not exponent: range(S)
   at the training length fell about 3x (138 -> 48), and growth over 4x length fell from 3.99x to
   3.32x. Music's local statistics are not the training marginal, so a residual drift survives
   centring. The intervention is therefore "much smaller excursion", not "bounded accumulator", and
   the account is supported at that strength only.
2. **Centring costs in distribution, detectably.** 0-512 NLL rises 0.524 -> 0.656 for MapPoPE and
   0.538 -> 0.648 for MapWM (both 0/5 seeds better). A sceptic's reading is available and is not
   excluded by this run: a smaller accumulator is a weaker positional signal, so the model leans on
   content, which costs resolution where position is reliable and helps where it is not. That story
   and the account make the same prediction here; separating them needs a manipulation that shrinks
   the OUT-OF-RANGE excursion without shrinking the in-range resolution, which centring does not do.
3. **MapPoPE still loses to MapWM under matched treatment.** Centred against centred: +0.008 in
   distribution (3/5), +0.065 at 1-2x (0/5, detectable), **+0.169 at 2-4x (0/5, detectable)**. So
   bounding the accumulator rescues MapPoPE from catastrophe but does not make the combination better
   than path integration alone -- consistent with the account, which says PoPE gives up the pairwise
   phase and gets a cleaner kernel in exchange, not a free win.

> **Scope (2026-09-20)**: the measurement in this file stands. The cross-task account it was read
> as supporting does not -- both halves fail a within-task test on Dyck (`T2_RESULTS.md`), so
> treat this as a fact about Bach at a 512-token context.

## Where this leaves the theory

The conjunction account stands: the collapse needs BOTH an out-of-range accumulator and the absence
of a pairwise phase to absorb it. Removing (most of) the first rescues MapPoPE specifically, 5.7x
more than it helps the arm that has the second. Rank and omega base failed to explain it because
neither touches either factor -- they reparameterise the angle, they do not change what the
accumulator measures or what can compensate for it.

Untested and now the sharpest remaining test: **T3**, a per-token `delta_c(x)`, which should recover
extrapolation WITHOUT the in-distribution cost that centring imposes -- and should pay for it in
PoPE's pure-indexing advantage instead. That double-sided prediction is what would separate the
account from caveat 2's sceptical reading.

## Batch provenance (audit note, 2026-09-19)

The contrasts here pair by seed index ACROSS run directories built on different days, not within one
batch -- the repo's standing rule 3 asks for one batch. Mitigating: the arms share code (only new
classes were added between the runs; the data and evaluation paths are byte-identical), the data
stream is seeded identically, and the primary readings lean on a parameter-matched inert twin
trained INSIDE the new batch. Unmitigated for the decay arms, which have no same-batch baseline.
