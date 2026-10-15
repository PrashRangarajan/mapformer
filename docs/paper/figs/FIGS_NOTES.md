# Figure notes (2026-10-14; post hoc, CPU, eval-only)

Scripts, outputs and figures in this directory. Run from `/home/prashr` with `PYTHONPATH=/home/prashr`.

## Figure 3: the mechanism (`fig_mechanism.py` -> `fig_mechanism.{png,pdf}`, numbers `fig_mechanism_out.txt`)

Data: MECHANISM batch (`MECHANISM_RESULTS.md`; ring 32, p_plus 0.5, T = 128, 300 epochs, seeds 800-807, n = 8 per arm).
Start = `eval.json` `init_angles`; end = `model_wrap_exist.action_angles` of the checkpoint (omega * step). 64 channels x 8
seeds per panel. Init-to-end segments are drawn for seed 800 only.

**View.** Each angle has its whole turns removed toward zero: rep(a) = sign(a) (|a| mod 2 pi). Adding 2 pi to one move's
angle does not change the model's function, so rep(a) is a valid representative, and it keeps the raw sign, which is what
the non-negative constraint acts on. Neither plain wrapped view works: in (-pi, pi]^2 and in [0, 2 pi)^2 every cancelling
pair falls on ONE line, so the two routes cannot be told apart. In this view the cancelling pairs lie on two lines:
a(-1) = -a(+1) through the origin (needs opposite raw signs, so the |.| step cannot reach it) and a(+1) + a(-1) = 2 pi
(or -2 pi). Signed raw angles are all within (-2 pi, 2 pi), so rep is the identity for them. AO_high's raw angles reach
17.7 rad (up to 3 turns), and its segments are skipped where a turn was removed. The through-zero / by-wrap split
describes where the parameters sit. It is not a property of the function (rule 8). The cancelling share itself is
wrap-invariant.

**Caption (draft).** Move angles (a(+1), a(-1)) of every rotary channel, at the start (grey) and after training
(dots). There are 8 runs per panel, on the 1D ring. A channel cancels opposite moves when a(+1) + a(-1) = 0 mod 2 pi, on the
blue line (through zero; reached only with opposite signs) or the red line (by wrapping). Grey shading lies outside the
non-negative quadrant, where the |.| step cannot go; the dotted line is a(+1) + a(-1) = pi; dashed boxes are the U(lo, hi)
start boxes. The panels:
- Signed step from the same-sign start (SO_absmag): 7/8 runs solve. One move's angle crosses zero: 6% of channels cancel
  at the start, 89% at the end (88% with opposite signs, none by wrapping).
- The |.| step from the same start (AO_absmag): 0/8 runs solve, with 4% cancelling.
- The |.| step from a large start whose sums stay below pi (AO_mid): 2/8 runs solve, with 5% cancelling, all by wrapping.
- The |.| step from a large start whose sums exceed pi (AO_high): 8/8 runs solve, and cancelling rises from 7% to 23%,
  all by wrapping.

The cancelling share is the registered wrap share: |wrap(a(+1) + a(-1))| < 0.1 rad among channels with |wrap(a(+1))| > 0.05.

Notes:
- AO_absmag's registered 0.04 is mostly not cancellation. 0.035 of it is same-sign channels near the origin (both angles
  < 0.1 rad), and 0.007 is by wrapping. Also, 44% of AO_absmag's channels end with |a(+1)| <= 0.05: their step collapses
  to ~0. The figure colours the near-origin channels black.
- The |.| start is not uniformly "sums below pi". |default| angles reach 5.5 rad, and initial sums run 0.01-6.58. AO_high's
  start already has 7% cancelling channels by chance, since iid U(1.7, 4.5) sums pass near 2 pi.

## Ring strata (plan item 7) (`ring_strata.py` -> `ring_strata.json`, `ring_strata_out.txt`; `fig_ring_strata.py` -> `fig_ring_strata.{png,pdf}`)

**Stream.** `GridWorldCancel(32, seed 10000, p_plus 0.5)`, `np.random.seed(10**6)`, maps redrawn per trajectory, T = 128
moves, eval mode with no dropout re-score. It uses **400** trajectories, not the 200 asked for: the 400 are the registered
eval stream, the first 200 are identical to a 200-trajectory eval, and using 400 makes the check below exact.

**Strata.** Every stratum is a revisit, and the gap is the number of moves since the last visit to the cell. A revisit is
"wrap-only" when every earlier visit is at unwrapped displacement +-32k with k != 0. Unwrapped position = cumsum of the move
tokens.

**Verification.**
- Constructed cases pass (`--test`): a full lap, the return, back-and-forth moves, laps in both directions, and runs of
  laps.
- The revisit flag equals `environment_cancel`'s mask on 50 random trajectories plus all 400 eval trajectories, and the
  unwrapped walk matches the environment's cells.
- Overall revisit accuracy reproduces every run's `eval.json` acc(128) (max |diff| 5e-5).

**Floor.** The constant predictor scores 0.49-0.50 in every stratum, so the strata do not differ in target composition.

| batch | arm | SOLVED | plain gap 2-7 | plain 8-31 | plain 32-127 | wrap-only | plain, wrap-gap-matched | all |
|---|---|---|---|---|---|---|---|---|
| (stream) | n targets | | 12183 | 17420 | 5110 | 4134 | | 38847 |
| RING_DIAL | Signed_r4 | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RING_DIAL | Abs_r4 | 2/8 | 0.992 (0.97-1.00) | 0.975 (0.93-1.00) | 0.957 (0.91-1.00) | 0.923 (0.80-1.00) | 0.956 | 0.973 |
| WRAP_EXISTENCE | SO | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WRAP_EXISTENCE | AO | 0/8 | 0.961 (0.89-1.00) | 0.916 (0.85-0.97) | 0.881 (0.81-0.93) | 0.872 (0.80-0.93) | 0.878 | 0.921 |
| MECHANISM | SO | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MECHANISM | AO | 0/8 | 0.978 (0.92-1.00) | 0.918 (0.85-1.00) | 0.873 (0.77-1.00) | 0.811 (0.75-0.88) | 0.874 | 0.919 |
| MECHANISM | AO_high | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Entries are means over runs, with the range in brackets. "Plain, wrap-gap-matched" re-weights plain revisits to the
wrap-only gap distribution (8-move bins over gaps 32-127).

**Answer.**
- The non-negative models fail mainly with distance. Accuracy falls with the gap since the last visit in every |.| batch:
  from gap 2-7 to gap 32-127 it drops by -0.035 for Abs_r4 (6/8 runs lower; the 2 solved runs sit at 1.000; sign-flip
  p 0.031), -0.080 for WRAP_EXISTENCE AO (8/8 lower, p 0.0078, the minimum for n = 8) and -0.105 for MECHANISM AO (8/8,
  p 0.0078). Short-gap revisits are near ceiling but not at it (0.96-0.99). This fits a code that works locally and drifts
  with distance.
- Wrap-only revisits add a deficit beyond the gap that varies between batches. Against gap-matched plain revisits it is
  -0.063 for MECHANISM AO (6/8 runs, p 0.031), -0.007 for WRAP_EXISTENCE AO (2/8, p 0.88) and -0.032 for Abs_r4 (4/8,
  p 0.31).
- The two AO batches are the same arm on different seeds, and they disagree. One MECHANISM AO run (s804) is a clean case
  of "local clock, no wrap": 1.000 / 0.999 at short / long plain gaps, 0.881 on wrap-only revisits.
- So the post hoc reading is: |.| models lose accuracy with gap in every batch, and the wrap-specific failure is
  real in some runs but not consistent. Signed models are at 1.000 in every stratum.
- AO_high is at 1.000 in every stratum (8/8 SOLVED). It "fixes" the long-gap and wrap-only strata, but at ceiling this is
  the same fact as SOLVED, not separate evidence.

**Caveats.**
- Post hoc and descriptive. The sign-flip p values are across runs within an arm, uncorrected for the 3 batches x 2 contrasts.
- Accuracy tracks loss (rule 2), so the strata describe where the unsolved models' errors fall; they do not establish a
  mechanism.
- Gap and wrap-only are confounded with each other by construction: wrap-only gaps are >= 32. That is why the gap-matched
  column exists.

**Caption (draft).** Revisit accuracy on the 1D ring by stratum (eval mode; 400 fresh trajectories, T = 128; maps redrawn).
"Plain" revisits have an earlier visit at the same unwrapped position. "Wrap-only" revisits were seen only a lap away
(+-32k). Lines show arm means; dots show runs; hollow markers show plain revisits re-weighted to the wrap-only gap
distribution. Signed steps (24 runs over three batches) and |.| steps started past pi (AO_high) are at 1.000 everywhere.
|.| steps from the default start lose accuracy with the gap (0.96-0.99 -> 0.87-0.96). Their extra wrap-only deficit
varies between batches (-0.063, -0.007, -0.032). The constant-predictor floor is 0.50.
