# SEARCH -- results (pre-registration: `SEARCH_PREREG.md`, commit 3a9e49a)

**Status: complete (S1, S2, S3 landed 2026-09-11).** Nothing in flight.

## Headline

1. **From-scratch EM DOES find the rewind: per query token, wrapped, for about half the k.**
   The "0/40" came from a readout that cannot see a wrapped solution (details below).
2. **The size of a rewind is not the obstacle.** When every query shares one k, EM finds a
   63-symbol rewind on 7/8 seeds (k = 64, mean 0.985) and a 15-symbol one on 8/8. It gets there
   faster than WM (median 54 against 99 epochs to loss < 0.5). At 2x length it beats WM,
   **+0.191, 7/8, detectable** (exploratory). The linear ratio is +0.3 to +1.4 against a target
   of -63 on every seed, so every found rewind is wrapped.
3. **A k curriculum helps and does not close the gap:** +0.127 (MDE 0.116, 7/8), 0/8 seeds
   >= 0.9, linear slope ~0. The whole gain is at k <= 32 (+0.300, +0.301, both detectable).
   The query tokens introduced last (k 33-64, from epoch 120-150) gain nothing.
4. **So the search problem is SPREAD, not size.** One k = one token that sees every query:
   found. Sixty-four k = sixty-four independent tokens, each seeing 1/64 of the queries: each
   finds its wrapped rewind or not. Account (untested): a token must find its rewind in the
   window before the kernel sharpens. S2 shows the path is smooth at init and rugged after
   training, and the curriculum helped only the tokens it introduced early.

## Headline of S1 and S2

**"EM never finds the rewind from scratch (0/40)" is wrong as stated.** It was read off a LINEAR
slope statistic. The rewind is only defined modulo each block's period (`theta` enters through
cos/sin), and from-scratch EM finds it that way: per query token, wrapped, for a subset of k.
In every EM arm, **every** solved large-k cell (k >= 8) goes through the query token's own step.
About 40% land on the kernel's peak, with A_X > 0 at the answer. About 55% land on its trough,
with A_X < 0; the two are the same mechanism under the sign gauge. Failed cells never carry it.
So the search problem is per token: each `q_k` either finds a wrapped rewind or does not.

## S1 -- retrieval anatomy (eval-only; 5 arms x 24 seeds, 256 held-out episodes each)

Deviation: the pre-registration says "128 episodes (~28 queries per k)". Those two numbers
disagree, because the scored rate is ~7 queries per 1024-token episode. 256 episodes were used,
to match the stated ~28 queries per k. `em_forward` reproduces `model(x)` on every checkpoint
(asserted, tolerance 1e-3). The recomputed accuracies agree with the stored evaluations, e.g.
AlignLock s8-12 0.682 / 0.743 / 0.403 / 0.507 / 0.742 against 0.701 / 0.737 / 0.396 / 0.492 / 0.782.

**H-wrap (S1-a): PARTIAL.** `VanillaEM_P0_r4`, cells with k >= 8:

| arm | cells | solved (acc >= 0.9) | solved with sel-sel0 >= 0.5 | failed (acc <= 0.3) | failed with it | r(acc, sel-sel0) |
|---|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 1368 | 595 | 0.459 | 433 | 0.000 | +0.460 |
| EMDoF_alignlock | 1368 | 595 | 0.395 | 496 | 0.000 | +0.440 |
| EMDoF_magonly | 1368 | 600 | 0.425 | 456 | 0.000 | +0.455 |
| EMDoF_alignfree | 1368 | 1009 | 0.445 | 283 | 0.000 | +0.382 |
| VanillaEM_r4 (sep) | 1368 | 958 | 0.522 | 274 | 0.000 | +0.422 |

The failed-cell half is met exactly (0.000 against <= 0.20). The solved-cell half is not
(0.459 against >= 0.70). The exploratory route table below shows why: the registered readout
looked only at the kernel's PEAK (argmax), and more than half the solved cells rewind to its
TROUGH instead.

**H-phase (S1-b): route prediction REFUTED; the manipulation check PASSES.**

- Every rho = 1 head peaks at symbol distance n = 0 (48/48 in P0, AlignLock and MagOnly), as
  `kappa(0) = sum a_i` requires. **All 48 AlignFree heads peak off zero**, at n = 2..63, and so do
  all 48 separate-q0/k0 heads (n = 4..78). So phase freedom is used, in every head.
- It does not change the ROUTE. The fraction of solved cells on the peak-rewind route is
  AlignFree 0.445 against MagOnly 0.425 (+0.020; <= -0.15 was predicted).
- Where the gain sits, AlignFree - MagOnly paired by seed (n=24):

| k bin | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| 1-16 | +0.124 | 0.147 | 0.084 | 21/24 | DETECTABLE |
| 17-32 | +0.263 | 0.192 | 0.110 | 22/24 | DETECTABLE |
| 33-48 | +0.077 | 0.232 | 0.133 | 15/24 | unmeasured |
| 49-64 | +0.130 | 0.247 | 0.141 | 17/24 | unmeasured |

### Exploratory (not registered): the complete route, and the distance account

Route of each solved cell (acc >= 0.9, k >= 8), read in the head with the most attention on the
answer. Categories in priority order: peak-rewind (sel - sel0 >= 0.5), trough-rewind
(selmin - selmin0 >= 0.5), peak-static (sel0 >= 0.5), trough-static (selmin0 >= 0.5).

| arm | solved | peak-rewind | trough-rewind | peak-static | trough-static | other | A_X > 0 on peak routes | A_X > 0 on trough routes |
|---|---|---|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 595 | 0.427 | 0.541 | 0.000 | 0.032 | 0.000 | 1.000 | 0.000 |
| EMDoF_alignlock | 595 | 0.371 | 0.576 | 0.000 | 0.045 | 0.007 | 1.000 | 0.000 |
| EMDoF_magonly | 600 | 0.398 | 0.557 | 0.000 | 0.045 | 0.000 | 1.000 | 0.000 |
| EMDoF_alignfree | 1009 | 0.424 | 0.515 | 0.028 | 0.033 | 0.000 | 1.000 | 0.000 |
| VanillaEM_r4 (sep) | 958 | 0.501 | 0.432 | 0.035 | 0.031 | 0.000 | 1.000 | 0.000 |

- 93-97% of solved large-k cells are rewinds of the query token, split between peak and trough.
  The sign of A_X at the answer tracks the route perfectly (1.000 / 0.000). That is the gauge
  `(-A_X) (*) (-kappa)` from `AUDIT_2026-09-10.md` #5, now visible inside trained models: a
  rewind to the trough with a negative content gain is the same retrieval.
- **Distance from the kernel peak does not explain phase freedom.** d = min over heads of
  |k - n*_h|:

| arm | d 0-3 | d 4-7 | d 8-15 | d 16-31 | d 32-64 |
|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 0.972 (72) | 0.740 (96) | 0.458 (192) | 0.352 (384) | 0.470 (792) |
| EMDoF_magonly | 0.944 (72) | 0.833 (96) | 0.547 (192) | 0.333 (384) | 0.463 (792) |
| EMDoF_alignfree | 0.812 (298) | 0.742 (252) | 0.770 (365) | 0.753 (380) | 0.722 (241) |
| VanillaEM_r4 (sep) | 0.732 (313) | 0.688 (324) | 0.733 (442) | 0.723 (376) | 0.802 (81) |

  In rho = 1 arms, success falls with k, then partly recovers at d >= 32. That fits trough targets,
  which sit far from the peak. In free-phase arms it is flat at ~0.75. Phase freedom does not
  shorten the shift a token has to make. It raises the per-token success rate at every distance.

## S2 -- the gradient at initialisation (no training)

AlignFree and MagOnly are the same function as P0 at init by construction, so their
function-space numbers are identical to P0's. This is expected, not a bug.

- **Silent.** The position pathway's gradient is **1.7e-4 to 4.3e-4 of the content branch's**
  (8/8 seeds). rms A_P at init is 6-9e-4 against rms A_X 0.31-0.34. The score is
  `A_X (*) A_P`, so attention is nearly uniform and the pathway sits at a multiplicative saddle.
- **Directionless.** The rate at which gradient flow moves the linear rewind slope has
  |t| < 2 on 7/8 seeds, and is negative on 4/8. **H-rugged (ii): MET.** cos(g, dJ) between the
  loss gradient and the direction that raises the kernel at the answer is -0.042..+0.023.
- **Smooth at init, rugged after training.** Peaks along the straight path from the current
  `Delta(q_k)` to the exact linear rewind, counting those with prominence >= 1% of kappa(0):

| set | k <= 4 | k 8-16 | k 32 | k 60-64 | kernel at the rewind > kernel now, k >= 32 |
|---|---|---|---|---|---|
| init (all three arms) | 0 | 0 | 1 | 2 | 0.727 |
| trained P0 s0-7 | 0 | 2 | 8 | 15.5 | 1.000 |

  H-rugged (i): **NOT MET at init** (k 60-64 median 2, >= 5 was needed); **MET at trained P0**
  (15.5). Deviation: the registered strict count read float noise on flat paths: 15.5 "maxima"
  at k <= 4 in trained P0, where the path is flat at 0.993. It is kept in parentheses in
  `runs/search/S2_report.md`. The prominence count was added after the first run and before
  the verdicts were read.

**Reading S2 with S1.** At init every query token faces a smooth path to its rewind, but the
gradient that would carry it there is ~3000x smaller than the content branch's and has no
consistent direction. Once the kernel has amplitude, the gradient is real, but the path has
become multi-modal: at trained P0, k = 60-64 tokens face ~15 barriers, and for every k >= 32
the rewind still scores higher on the kernel than the point the token settled at. That is a
window problem. The landscape is smooth while it is silent and rugged once it can be heard.
It is consistent with S1's per-token, all-or-nothing successes. **This is an account, not a
test.** The S2 readouts are two snapshots (init and end); nothing here has tracked when the
barriers appear relative to when the gradient grows.

## S3 -- training (4 arms x 8 seeds, one batch; `runs/search/S3_report.md`)

**Determinism re-check PASSES:** `P0_repro` s0 is bitwise identical to the stored
`runs/dof/recency` P0 s0 (25/25 tensors, loss curves equal). The environment and trainer edits
left the default task unchanged, and reuse of the stored P0 s0-7 as the curriculum comparator
is licensed. (The first render of the report said "reuse NOT licensed". That was a string check
matching "differing 0". It is fixed and the report regenerated.)

| arm | acc T=1024 | >= 0.9 | >= 0.95 | acc T=2048 | final loss | epochs to loss < 0.5 (median, reached) |
|---|---|---|---|---|---|---|
| P0_fix64 (EM, k = 64) | 0.985 +/- 0.043 | 7/8 | 7/8 | 0.946 | 0.065 | 54 (8/8) |
| P0_fix16 (EM, k = 16) | 0.994 +/- 0.018 | 8/8 | 7/8 | 0.966 | 0.015 | 25 (8/8) |
| WM_fix64 (WM, k = 64) | 0.947 +/- 0.120 | 7/8 | 6/8 | 0.755 | 0.119 | 99 (7/8) |
| P0_cur (EM, curriculum) | 0.727 +/- 0.038 | 0/8 | 0/8 | 0.668 | 1.045 | 11 (8/8) * |

\* not comparable: the curriculum's first epochs ask only k <= 2.

Registered verdicts:
- **S3-P1 (positive control) MET**: WM_fix64 >= 0.95 on 6/8.
- **S3-P2 REFUTED**: P0_fix16 8/8 as predicted, but P0_fix64 is >= 0.9 on **7/8** (<= 2 was
  predicted). The size of the rewind does not make it unfindable. That also takes the weight
  off H-rugged: trained standard-task P0 paths are rugged, yet a single k = 64 token gets through.
- **S3-P3 MET, and the gain is detectable**: `P0_cur - P0` = +0.127 (sd 0.117, MDE 0.116, 7/8).
  P0_cur is under 0.9 on 8/8 seeds and its linear slopes are -0.015..+0.009. The curriculum
  helps. It does not find the linear rewind and does not close the gap.

Exploratory contrasts (paired by seed, n=8):

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EM - WM, fixed k = 64, T=1024 | +0.038 | 0.135 | 0.133 | 3/8 | unmeasured |
| EM - WM, fixed k = 64, T=2048 | **+0.191** | 0.152 | 0.150 | 7/8 | DETECTABLE |
| P0_fix16 - P0_fix64 | +0.009 | 0.049 | 0.048 | 1/8 | unmeasured |
| P0_cur - P0, k 1-16 | **+0.300** | 0.193 | 0.191 | 8/8 | DETECTABLE |
| P0_cur - P0, k 17-32 | **+0.301** | 0.223 | 0.221 | 6/8 | DETECTABLE |
| P0_cur - P0, k 33-48 | -0.080 | 0.177 | 0.175 | 4/8 | unmeasured |
| P0_cur - P0, k 49-64 | -0.091 | 0.234 | 0.232 | 2/8 | unmeasured |

The curriculum's k <= 32 tokens were introduced by epoch 90. The k 33-64 tokens came at
epochs 120 and 150, and only those show no gain.

Fixed-k mechanism, single query token (full table in `runs/search/S3_report.md`):
- **P0_fix16**: peak-rewind on 8/8 seeds (sel - sel0 = +1.000 on every seed). Linear ratios
  range from +0.97 to -15.04 against a target of -15. Two seeds found the linear rewind itself
  (-15.04, -14.80); six found a wrapped one. Idealised wrapped score 0.87-1.00.
- **P0_fix64**: peak-rewind on 6/8, trough-rewind on 1/8, none visible on 1/8 (s0, accuracy
  1.000 by a route these argmax readouts do not capture). Linear ratios +0.32..+1.42 against
  -63: **0/8 linear, 7/8 wrapped.** The idealised wrapped score is only -0.84..+0.86 even on the
  peak-rewind seeds. That score assumes filler Delta = 0 and one shared symbol Delta, so the
  models evidently compensate through filler. `sel - sel0` is the readout that measures the
  retrieval itself.

**Where this leaves the account.** The obstacle is neither representation (1.000 frozen),
nor stability (1.000 held), nor the size of one rewind (7/8 at k = 64). It is that the
standard task asks for 64 separate wrapped rewinds, one per query token, each trained by
1/64 of the queries. Candidate mechanism (not tested): each token must find its rewind before
the kernel sharpens and its landscape turns multi-modal (S2). Direct tests, cheapest first:
(a) the standard task with k drawn from a small set spanning 1..64 (4 or 16 values), holding
queries per token fixed. The account predicts that success tracks queries per token, not the
largest k. (b) Record per-token rewind status and path ruggedness every epoch from scratch,
to time the window.

**Neuron [11] context.** [11] predicts EM learns faster except on N-back. Here EM learns a
FIXED-offset k-back faster than WM (54 vs 99 epochs) and holds it at 2x length (+0.191). Its
deficit appears only when the offset varies per query. Exploratory, n=8, one task.
