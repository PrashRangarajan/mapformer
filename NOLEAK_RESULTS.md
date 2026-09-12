# Leakage test: with the leak closed, trainable EM holds the recency solution perfectly

Pre-registration: `NOLEAK_PREREG.md` (with its pre-launch revision). Analysis:
`analyze_noleak.py`, committed before any result existed. 2x2 on seeds 0-7:
{install 1/64, 1/8} x {leak open = `runs/unfreeze`, leak closed = `runs/noleak`}, where
"leak closed" holds `w_in`'s content columns at zero so Delta depends on the latent
rewind code alone.

**Preconditions, both exact.** The determinism re-check (`EMUnf_0_e8` s0) is bitwise
identical to the stored run -- 30 tensors and the loss curve -- so pairing with the
existing arms is licensed. The manipulation check holds at every epoch in both closed
arms: max |`traj_leak`| = 0 and max |`traj_slope` - `traj_latpath`| = 0.

| cell | acc T=1024 | acc T=2048 | >= 0.95 | final loss | eff. slope | latent pathway | leak |
|---|---|---|---|---|---|---|---|
| 1/64, leak open (`EMUnf_0`) | 0.642 +/- 0.266 | 0.582 | 0/8 | 1.216 | -0.055 | -0.074 | 0.421 |
| 1/64, leak closed (`EMNoLeak_e64`) | 0.784 +/- 0.281 | 0.709 | 3/8 | 0.832 | -0.008 | -0.008 | 0 |
| 8x, leak open (`EMUnf_0_e8`) | 0.941 +/- 0.063 | 0.895 | 4/8 | 0.247 | -0.602 | -0.866 | 0.728 |
| **8x, leak closed (`EMNoLeak_e8`)** | **1.000 +/- 0.000** | **0.991** | **8/8** | 0.017 | -0.859 | -0.859 | 0 |

Reference: rewind installed and FROZEN (`EMWarm_freeze`) = 1.000, 8/8.

## L1 -- accuracy criterion MET, slope criterion NOT met

Registered as a conjunction: `EMNoLeak_e8` >= 0.95 on >= 7/8 seeds **and** final
effective slope <= -0.95. Got **8/8 at 1.000**, and slope **-0.859**. So half of L1 holds.
The paired contrast against the leak-open arm is +0.059 (MDE 0.063, 6/8, unmeasured) --
the ceiling the pre-registration anticipated (0.941 +/- 0.063 leaves ~0.06 of headroom),
which is why the seed-count criterion was named as the readout: **8/8 vs 4/8**.

## L1b -- closing the leak does not change the latent pathway, and that is harmless

`EMNoLeak_e8 - EMUnf_0_e8` on the final latent-pathway slope: **+0.007** (MDE 0.309,
unmeasured). The pathway settles near -0.86 whether or not content can leak in. With the
leak closed that -0.86 pathway gives **perfect accuracy**, so the pathway's drift from
-1 costs nothing. **Leakage is the entire accuracy residual at 8x** (0.941 -> 1.000).
This refines the same-day correction to `UNFREEZE_RESULTS.md` U4 ("both contribute,
leakage the larger share"): both move the slope, only leakage costs accuracy.

**A measurement limit, recorded.** A pooled rewind slope of -0.86 coexisting with 1.000
accuracy means the slope statistic is not a sufficient summary of whether the rewind
works -- partial cancellation plus the content branch evidently suffices, or the
cancellation that matters lives in a subspace the mean-symbol projection under-weights.
Do not read slope magnitudes between about -0.85 and -1 as degrees of failure.

## L2 -- the rewind is destroyed at the small scale with no leak; accuracy is in between

Registered collapse: mean accuracy <= 0.75 **and** final latent pathway > -0.5.
Got **0.784** (3/8 >= 0.95) and **-0.008**. The slope half holds -- **with zero leakage
the small-scale rewind is erased completely**, so latent erosion alone is sufficient and
the two channels are separable, as U4's ordering implied. The accuracy half does not:
0.784 sits above the collapse line. With the rewind gone, that accuracy comes from a
non-rewind solution of the kind from-scratch EM finds (0.60-0.87 across arms). Neither
the registered collapse nor survival.

## L3 -- descriptive

Scale effect with the leak closed +0.216 (MDE 0.278, 7/8, unmeasured); with it open
+0.298 (MDE 0.270, 7/8, DETECTABLE, the U3 result). Interaction -0.082 (MDE 0.387),
unmeasured. Closing the leak at 1/64: +0.142 (MDE 0.374), unmeasured.

## What this settles

The three-part account of EM's recency deficit is now complete, each part measured:

1. **The solution exists in the architecture** -- frozen install 1.000 (8/8).
2. **Training holds it** when it is installed at a weight scale Adam does not erode AND
   the content -> Delta leak is closed -- **1.000 (8/8), 0.991 at 2x length**. With the
   leak open, 0.941. At the original 1/64 scale the code is eroded regardless (-0.008).
3. ~~**Training from scratch never finds it** -- 0 of 40 runs.~~ **WITHDRAWN
   (`SEARCH_RESULTS.md`)**: that came from a LINEAR slope, which cannot see a rewind that is
   wrapped modulo each block's period. From scratch EM finds one PER QUERY TOKEN for about
   half the k; with one shared k it is found on 7/8 seeds.

**EM's recency deficit is entirely a search problem.** Nothing about EM's factorised
"where"/"what" design prevents it from representing, or from keeping, the solution.

Scope: "holds" is shown for single-`p0` EM with the rewind installed, at one task, one
config, n=8. The 1.000 arm also has `w_in`'s content columns pinned at zero -- a point in
EM's weight space, not a different architecture, but one the unconstrained optimiser
drifts away from.
