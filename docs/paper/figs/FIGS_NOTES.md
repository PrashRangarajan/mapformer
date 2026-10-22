# Figure notes (2026-10-14; updated 2026-10-16; post hoc, CPU, eval-only)

Scripts, outputs and figures in this directory. Run from `/home/prashr` with `CUDA_VISIBLE_DEVICES=""
PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/<script>`. Every `fig_<name>.py` writes `fig_<name>.{pdf,png}`
and `fig_<name>_out.txt` (the numbers drawn). Shared style: `figstyle.py` (dataviz reference palette, light mode: slot 1
blue `#2a78d6`, slot 2 orange `#eb6834`, slot 3 aqua `#1baf7a`; these three validate all-pairs for colour-vision
deficiency; a fourth series (yellow `#eda100`) appears only where markers / legends also carry identity). Within a figure,
signed / path integration is blue and |.| / index RoPE is orange. `fig_mechanism` and `fig_ring_strata` keep their older
matplotlib default colours (only their content was fixed). The palette validator (`validate_palette.js`) was not run
(no node on this machine); the slots used are the reference palette's documented passing order.

**Status labels.** "Registered" = the batch and its verdict / class were pre-registered; the figure only re-plots them.
"Post hoc" = the quantity was computed after the results were read (floors, usage weights, strata re-cuts, plots of
per-epoch losses). Each figure is a post hoc DRAWING; the label in each caption says what the drawn numbers are.

## Index (2026-10-16)

| figure | outline | shows | status of the numbers |
|---|---|---|---|
| `fig_loss_curves` | F4 (headline) | training loss per epoch, signed vs |.| (softplus), 4 batches | registered batches and classes; plot post hoc |
| `fig_loss_curves_rank` | F4 | training loss per epoch, rank 2 / 3 / 4 | registered batches and classes; plot post hoc |
| `fig_wrap_share` (redrawn) | F7 / draft-5 Fig. 1 | cancelling share, usage-weighted (primary) + unweighted | post hoc |
| `fig_mechanism` (shading fixed) | F8 / draft-5 Fig. 2 | move-angle planes, ring | registered batch; view post hoc |
| `fig_factorial_strata` (floor fixed) | F2 part / Fig. K1 | torus strata, index depth / loop | registered secondary; floors post hoc |
| `fig_strata_modes` | F2 | torus strata for index / rank / sign failures | registered strata; floors post hoc |
| `fig_exchange_rate` | F1 | accuracy vs index layers, path 1 layer, Dyck / ring / torus | registered; floors post hoc |
| `fig_rank_solved` | F3 | SOLVED by per-head rank; rank = D vs D + 1 | registered |
| `fig_existence` | F5 | from scratch vs installed (frozen / trained) | registered |
| `fig_textworld` | F6 | text world: cancellation, clock, ablation | registered batch, declared secondaries |
| `fig_ring_strata` (floor added) | F9 | ring strata | post hoc |
| `fig_segments` (2026-10-21) | plan 8 / draft-2 Fig. 4 | accuracy vs direction runs spanned, torus index depth / loop, ring index depth | registered batches; bins post hoc (analysis) |
| `fig_clock_endstate` (2026-10-21) | plan 9 / draft-2 Fig. 5 | clock / map per run, one definition, by arm | registered batches and classes; ratio post hoc (analysis) |
| `fig_kernel_peak` (2026-10-21) | plan 11 / draft-2 Fig. 10 | kernel peak offset, C vs C0, C0 by stratum | registered batches; probes post hoc (analysis) |

Floor helpers (post hoc, CPU, no model): `torus_strata_floors.py` -> `torus_strata_floors_s500-503.json` (FACTORIAL's
stream) and `torus_strata_floors_s0-7.json` (the stream of SIGN_MATCHED / RANK_MI / RANK3 / RANK_SEP, seeds 0-7): per
seed and stratum the best constant (asserted equal, n and value, to every `*_STRATA.json` 'floor' of those seeds), blank
and retrace. Its overall retrace reproduces `docs/audits/2026-10-16/sign_floors_out.txt` per seed (0.8442, 0.8403, ...).
`ring_strata_floors.py` -> `ring_strata_floors.{json,_out.txt}`: the same per stratum on the ring strata stream.

**Not drawn anywhere:** the torus T = 128 retrace floor 0.934 (`paper2x2_T128_floors_out.txt`) -- no figure draws T = 128
torus accuracies (PAPER2X2 is a table). Where floors are drawn, the caption names which.

## Headline: training-loss curves, signed vs |.| (`fig_loss_curves.py`; outline F4)

Data: the `losses` array (per-epoch mean training loss, dropout on) of every stored checkpoint:
(a) SIGN_MATCHED `runs/sign_matched/p0/{Signed,Abs,Pos}_r4_s0-7` (torus T = 1024, 900 epochs, 1 layer d 128, lr 1e-3
cosine); (b) WRAP_EXISTENCE `runs/wrap_existence/p0/{SO,AO}_s600-607` (ring 32 cells, T = 128, 300 epochs,
observation-step-free, from scratch); (c) GENERALITY `runs/generality/torus/{SO4,AO4}/p0/*_s700-707` (torus T = 1024,
observation-step-free); (d) ROBUST `runs/robust/torus/{SO,AO}/p0/*_s1400-1407` (as (c), 2 layers, d 256). Classes by
`stats_core.classify_run` (= the registered ones; counts match each results file).

Numbers (`fig_loss_curves_out.txt`): SOLVED and class counts, median loss at the last epoch.

| panel | signed | |.| | softplus |
|---|---|---|---|
| (a) SIGN_MATCHED | 8/8 SOLVED; median 2.8e-3 | 0/8 (5 STALLED, 3 DESCENDING); median 0.47 | 1/8 (s7, final-5% 0.049; 4 STALLED, 3 DESC.); 0.67 |
| (b) WRAP_EXISTENCE ring | 8/8; ~1e-5 | 0/8 (8 STALLED); 0.28 | -- |
| (c) GENERALITY torus | 8/8; 2.9e-3 | 0/8 (6 STALLED, 2 DESC.); 0.43 | -- |
| (d) ROBUST torus 2L d256 | 8/8; 6.3e-4 | 2/8 (s1401, s1404; 6 DESCENDING); 0.081 | -- |

Epoch at which the median first falls below 0.05: signed 480 / 57 / 309 / 183 (a-d); |.| and softplus never.
Median loss at 10% / 25% / 50% / 75% of training: (a) signed 1.21 / 0.23 / 0.075 / 0.025, |.| 1.54 / 1.16 / 0.56 / 0.50;
(b) signed 0.37 / 0.019 / 8e-4 / 2e-5, |.| 1.04 / 0.60 / 0.37 / 0.30; (c) signed 0.89 / 0.10 / 0.023 / 0.006, |.| 1.56 /
0.66 / 0.55 / 0.47; (d) signed 0.15 / 0.047 / 0.013 / 0.003, |.| 0.65 / 0.39 / 0.24 / 0.15.

**Caption (ready to paste; registered batches and classes, post hoc plot).** Training loss per epoch for the signed step
(blue) and the non-negative steps |.| (orange) and softplus (green), every run of four registered batches (thin lines;
thick: median of 8 runs; log scale; training loss with dropout on). Dashed line: the registered SOLVED threshold
(final-5% mean training loss < 0.05); n/8 = runs SOLVED. (a) Torus, T = 1024, one layer (SIGN_MATCHED): signed 8/8, |.|
0/8, softplus 1/8. (b) Ring of 32 cells, observation-step-free (WRAP_EXISTENCE): 8/8 vs 0/8. (c) Torus,
observation-step-free (GENERALITY): 8/8 vs 0/8. (d) As (c) at two layers and d_model 256 (ROBUST): 8/8 vs 2/8. The signed
median crosses the threshold at epoch 480 / 57 / 309 / 183 of 900 / 300 / 900 / 900 and keeps descending; the |.| median
never does, and most |.| runs flatten at 0.3-0.5 (STALLED on the ring 8/8; 5-6/8 on the 1-layer torus). The 2-layer |.| runs are still descending at 900 epochs (6/8), so
there part of the cost may be speed (budget-scoped).

Caveats: training loss is dropout-on (REPRO note 7: a run can score ~1.000 held out while missing the loss cut). Held-out
T = 1024 accuracy of the unsolved non-negative runs (stored eval JSONs): SIGN_MATCHED |.| 0.64-0.93, softplus 0.53-0.97;
GENERALITY |.| 0.71-0.96; ROBUST |.| 0.91-0.994 (s1400 0.994 is unsolved by loss but near ceiling held out -- at 2 layers
the loss curve and the accuracy disagree most). The softplus SOLVED run (s7, 0.995 held out) sits at 0.049, on the
threshold.

## Training-loss curves by rank (`fig_loss_curves_rank.py`; outline F4, rank half)

Data: `runs/rank_mi/p0/{Vanilla,Vanilla_r4mi,Vanilla_r2ph}_s0-7`, `runs/rank3/p0/Vanilla_r3ph_s0-7`,
`runs/rank_sep/p0/Vanilla_r4ph_s0-7` (torus T = 1024, 900 epochs; every arm built from our r = 2's initial weights at each
seed). Numbers (`fig_loss_curves_rank_out.txt`): (a) shared r 2 0/8 SOLVED (4 STALLED, 4 DESCENDING; median final 0.35)
vs shared r 4 8/8 (2.8e-3); (b) per-head r 2 2/8 (median 0.22), r 3 6/8 (3.5e-3), r 4 8/8 (3.1e-3).

**Caption (registered batches and classes, post hoc plot).** Training loss per epoch on the torus (T = 1024) by the
per-head rank of the content-to-angle map; thin lines are runs, thick lines medians of 8; dashed: the SOLVED threshold.
(a) One latent shared by the heads: rank 2 solves 0/8, rank 4 8/8 (RANK_MI). (b) Per-head latents: rank 2 2/8, rank 3 6/8,
rank 4 8/8 (RANK_MI, RANK3, RANK_SEP). The rank-4 and rank-3 medians cross the threshold at epochs 478-532; the rank-2 medians never
do, and most rank-2 runs flatten at 0.2-0.5.

## Fig. 1 of draft 5, redrawn: cancelling share, usage-weighted (`fig_wrap_share.py`; outline F7)

Redrawn 2026-10-16 for MOCK_REVIEW_draft5 W4 / SYNTHESIS item 9. Same runs as before: RING_DIAL (`runs/ring_dial/p0`,
ring 32, T = 128, 300 epochs, seeds 30-37, p_plus 0.5 / 0.75 / 0.9) and SIGN_MATCHED (torus T = 1024, seeds 0-7).

**The measure.** Entries are (move, channel) pairs, a = omega_c * step_c(move). Selected: |wrap(a)| > 0.05 rad. **The
denominator excludes channels whose move angle has collapsed to ~0 mod 2 pi** (the old caption's "share of rotary
channels" was wrong; the share is over non-collapsed channels). Cancelling: |wrap(a(move) + a(opposite))| < 0.1 rad;
torus pairs (0,1), (2,3). Primary = usage-weighted: each entry weighted by mean |q_c| at move positions x mean |k_c| at
observation positions (`model_wrap_exist.usage_weights`, ACCOUNTS Amendment 1; inlined for VARIANT_MAP models), on 32
eval walks (ring: GridWorldCancel(32, seed 10000, p_plus as trained), np seed 10**6, T = 128; torus: GridWorld(64, seed
10000), np seed 1234, T = 1024). Secondary (faint dash) = unweighted. The unweighted values reproduce the old figure
exactly (0.79 / 0.06 / 0.78 / 0.03 / 0.64 / 0.06 / 0.98 / 0.11 / 0.03).

| cell | arm | SOLVED | weighted mean (range) | unweighted mean (range) | excluded (collapsed) entries |
|---|---|---|---|---|---|
| ring p_plus 0.5 | signed | 8/8 | 0.79 (0.41-0.99) | 0.79 (0.59-0.92) | 0.11 |
| ring p_plus 0.5 | |.| | 2/8 | 0.06 (0.00-0.29) | 0.06 (0.02-0.16) | 0.29 |
| ring p_plus 0.75 | signed | 8/8 | 0.87 (0.55-0.99) | 0.78 (0.64-0.86) | 0.11 |
| ring p_plus 0.75 | |.| | 1/8 | 0.02 (0.00-0.07) | 0.03 (0.02-0.04) | 0.24 |
| ring p_plus 0.9 | signed | 8/8 | 0.70 (0.39-0.99) | 0.64 (0.44-0.91) | 0.16 |
| ring p_plus 0.9 | |.| | 4/8 | 0.10 (0.00-0.34) | 0.06 (0.02-0.10) | 0.18 |
| torus T 1024 | signed | 8/8 | 0.93 (0.64-1.00) | 0.98 (0.90-1.00) | 0.35 |
| torus T 1024 | |.| | 0/8 | 0.09 (0.00-0.29) | 0.11 (0.00-0.27) | 0.68 |
| torus T 1024 | softplus | 1/8 | 0.01 (0.00-0.09) | 0.03 (0.00-0.12) | 0.62 |

Solved non-negative runs, weighted: ring 0.01, 0.01 (p 0.5); 0.02 (p 0.75); 0.34, 0.01, 0.07, 0.01 (p 0.9); torus softplus
0.00. Per-run values in `fig_wrap_share_out.txt`.

**Reading.** Weighting by use does not change the picture: signed means 0.70-0.93, |.| 0.02-0.10, softplus 0.01. It
widens the signed spread (some signed runs read more of their non-cancelling channels: 0.39-0.64 minima), and 7 of the 8
solved non-negative runs stay at <= 0.07 weighted, so solved |.| runs do not use the wrapped code in the channels attention
reads either. (ACCOUNTS T1's 0.87 -> 0.18 vs 0.99 -> 0.98 gap was for a held CONSTRUCTED solution; on these from-scratch
runs the two measures agree to within ~0.1 on the means.)

**Caption (ready to paste; post hoc).** Share of rotary channels in which opposite moves cancel mod 2 pi, per run,
weighted by how much attention uses each channel (dots; filled = SOLVED) and unweighted (faint dashes, joined to the same
run). A channel cancels when |wrap(a(move) + a(opposite move))| < 0.1 rad; channels whose move angle has collapsed to
within 0.05 rad of 0 mod 2 pi are excluded from the denominator. Usage weight: mean query norm of the channel at move
tokens times mean key norm at observation tokens (eval walks). Ring of 32 cells at three move biases p_plus (RING_DIAL,
n = 8 each) and the torus at T = 1024 (SIGN_MATCHED, n = 8). Bars: arm means of the weighted share. Signed steps: 0.70-0.93;
|.| steps: 0.02-0.10; softplus: 0.01. Seven of the eight non-negative runs that solve stay at or below 0.07.

## Fig. 2 of draft 4 (Figure 3 of the outline): the mechanism (`fig_mechanism.py` -> `fig_mechanism.{png,pdf}`, numbers
`fig_mechanism_out.txt`)

**Shading fixed 2026-10-16** (MOCK_REVIEW_draft5 W9): omega is free in sign, so the |.| class (omega * |s|) contains both
the all-positive and the all-negative quadrant (appendix E P3); only the two MIXED-sign quadrants (II and IV) are closed to
|.|. The figure now shades only those, labelled "mixed signs: closed to |.|". No data moved; `fig_mechanism_out.txt` is
byte-identical (checked with cmp). No |.| point lies in quadrant III (raw ranges >= 0), so the old figure's data were right
and only the shading misstated the class.

Relabelled 2026-10-15: panel titles use draft 4's terms ((a) signed step, same-direction start = SO_absmag; (b) |.| step,
the same start = AO_absmag; (c) |.| step, large start, sums < pi = AO_mid; (d) |.| step, start past pi = AO_high); the
shading is labelled as the barrier side (closed to |.|). Numbers unchanged (`fig_mechanism_out.txt` byte-identical).

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
blue line (through zero; reached only with opposite signs) or the red line (by wrapping). Grey shading marks the two
mixed-sign quadrants, the only region the |.| step cannot reach (omega's sign is free, so both angles may be negative); the dotted line is a(+1) + a(-1) = pi; dashed boxes are the U(lo, hi)
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

**Floor added 2026-10-16 (post hoc; `ring_strata_floors.py`, the same 400-walk stream).** Per-stratum retrace (one-level
rule of `docs/audits/2026-10-16/ring_floors.py`): gap 2-7 **0.9998**, gap 8-31 **0.943**, gap 32-127 0.570, wrap-only 0.493
(blank 0.49-0.50 everywhere; overall retrace 0.864 on this stream vs 0.860 on ring_floors' stream). Drawn as dotted dashes
where on axis; the off-axis values are printed in the panel. **Consequence:** in the short-gap strata a map-free rule is at
or above the unsolved |.| arms (gap 2-7: retrace 1.000 vs |.| 0.96-0.99; gap 8-31: 0.943 vs WRAP_EXISTENCE AO 0.916 and
MECHANISM AO 0.918; RING_DIAL Abs_r4 0.975 is above). The |.| models' advantage over map-free rules is in the long-gap and
wrap-only strata (0.81-0.96 vs retrace 0.57 / 0.49). "Works locally" in the answer above therefore means "about as well as
undoing the last run", not "better than a map-free rule".

**Caption addition (post hoc).** Dotted dashes: the retrace floor within each stratum (copy the observation seen when the
walk last undid its most recent run of moves): 1.000 at gaps 2-7 and 0.943 at gaps 8-31; 0.57 and 0.49 in the long-gap and
wrap-only strata (off axis).

## Fig. K1 fixed: torus strata, index depth and loop (`fig_factorial_strata.py`)

Fixed 2026-10-16 (NUMBERS_AUDIT item 1). The old dashed line was the mean of the per-stratum constant floors over all arms
and runs (recomputed: 0.513), while the caption said "overall 0.507" (a number from another stream, RANK_MI seeds 10-25).
Now each stratum carries its own floors, computed on FACTORIAL's own eval stream (seeds 500-503; `torus_strata_floors.py`,
asserted equal in n and value to the JSON 'floor' of every FACTORIAL run): best constant (= blank) 0.508 / 0.515 / 0.517
(short gap / long gap / wrap-only), and retrace 0.912 in the short-gap stratum (retrace equals the constant in the other two:
the walk's runs are at most 10 moves, so a retrace never reaches a gap >= 128 or a wrap-only revisit). Overall on this
stream: constant 0.509 (0.505-0.515), retrace 0.844 (0.840-0.848).

Numbers (`fig_factorial_strata_out.txt`; mean (range), n = 4): path 1L 1.000 / 1.000 / 1.000; index 1L 0.769 (0.735-0.787)
/ 0.480 (0.456-0.508) / 0.482 (0.470-0.495); index 4L 0.943 (0.900-0.958) / 0.458 (0.367-0.506) / 0.457 (0.381-0.494);
looped x4 0.938 (0.937-0.940) / 0.507 (0.496-0.531) / 0.509 (0.496-0.521).

**Caption (ready to paste; registered strata secondary; floors post hoc).** Revisit accuracy by stratum on the torus at
T = 1024 (FACTORIAL; trained and tested at T = 1024; bars: means of n = 4 runs, dots: runs). Strata: plain revisits with a
gap of fewer than 128 moves since the last visit, plain revisits with longer gaps, and wrap-only revisits (the cell was
seen before only from a different unwrapped position). Dashed: always-blank (the best constant) within each stratum, 0.508 /
0.515 / 0.517; dotted: the retrace rule within the short-gap stratum, 0.912 (elsewhere it equals the constant). Floors are
computed on this batch's own evaluation walks. Path integration with one layer is at 1.000 in every stratum. Four index
layers or a loop of four lift short-gap revisits to 0.94, just above retrace; at long gaps and wrap-only revisits every
index model stays at or below the constant (0.46-0.51).

## F2: strata for every failure mode (`fig_strata_modes.py`)

Data: (a) `FACTORIAL_{L1,L4,LOOP}_STRATA.json` (seeds 500-503); (b) `RANK_MI_STRATA.json` (Vanilla = shared r 2,
Vanilla_r2ph, Vanilla_r4mi) and `RANK3_STRATA.json` (Vanilla_r3ph), seeds 0-7; (c) `SIGN_MATCHED_STRATA.json` (Signed_r4,
Abs_r4, Pos_r4, RoPE), seeds 0-7. All eval mode at T = 1024, held-out map. Floors per stratum from
`torus_strata_floors_s500-503.json` (a) and `torus_strata_floors_s0-7.json` (b, c).

Numbers (`fig_strata_modes_out.txt`; short gap / long gap / wrap-only means):
- (a) as Fig. K1.
- (b) r 4 shared 1.000 / 0.995 / 0.977; r 3 per-head 1.000 / 0.967 / 0.875; r 2 per-head 0.919 / 0.784 / 0.644; r 2 shared
  0.950 / 0.671 / 0.557.
- (c) signed 1.000 / 0.995 / 0.982; |.| 0.870 / 0.630 / 0.515; softplus 0.805 / 0.729 / 0.574; index RoPE 1L 0.784 / 0.462 /
  0.471.
- Floors, seeds 0-7: constant 0.506 / 0.500 / 0.507; retrace 0.911 (short gap only).

**Caption (registered strata; floors post hoc).** Revisit accuracy by stratum on the torus at T = 1024 for the three ways a
model fails to learn the map. Lines: arm means; dots: runs. (a) Index attention (FACTORIAL, n = 4): depth and a loop help
short-gap revisits only. (b) Per-head rank of the content-to-angle map (RANK_MI, RANK3; n = 8, same seeds and initial
weights): rank 4 is near 1.000 in every stratum; rank 3 loses wrap-only revisits on two runs (0.49, 0.63); rank 2 loses with gap and
most at wrap-only revisits. (c) Sign of the step (SIGN_MATCHED, n = 8): signed near 1.000 everywhere; |.| and softplus fall
from short gaps to wrap-only revisits; one-layer index RoPE for reference. Dashed: always-blank within each stratum;
dotted: retrace within the short-gap stratum (0.91). Floors on each batch's own walks.

Note: in the short-gap stratum the mean |.| (0.870), softplus (0.805) and one-layer index (0.784, 0.769) arms sit BELOW
retrace 0.911-0.912, and the rank-2 arms only just above it (0.919, 0.950); their advantage over map-free rules is in the
long-gap and wrap-only strata (where retrace = the constant, ~0.50-0.52).

## F1: the exchange rate (`fig_exchange_rate.py`)

Data: (a) `DYCK_MDEPTH_RESULTS.json` cells `T12|{RoPE,MapWM}|{1-4}L|L32D12|A2f_close_acc_dist_feasible` (registered;
trained and scored at L32 D12, 1x budget, n = 8); floor 0.5935 = `floors_A2f['T12']['L32D12']`. (b) `runs/cancel/p0/*/
eval.json` acc at T = 128, **fresh seeds 2-7 only** (n = 6; seeds 0-1 were the pilot; CANCEL_RESULTS CORRECTED block, and
SYNTHESIS item 4: the main text cites fresh-seed values); retrace floors 0.860 / 0.758 / 0.612 from
`docs/audits/2026-10-16/ring_floors_out.txt` (same task, another stream). (c) `FACTORIAL_{L1,L4,LOOP}.json` acc at 1024
(n = 4; --fast-attn); floors on its own stream (`torus_strata_floors_s500-503.json`).

Numbers (`fig_exchange_rate_out.txt`):
- (a) Dyck A2f: path MapWM 1L 0.996; index RoPE 1L 0.648, 2L 0.889, 3L 0.964, 4L 0.982; floor 0.594. (MapWM 2-4L 1.000.)
- (b) ring, fresh seeds: path 1L 1.000 at every p_plus; index 1 / 2 / 3 layers: p 0.5 0.710 / 0.949 / 0.998; p 0.75 0.678
  / 0.962 / 0.998; p 0.9 0.824 / 0.987 / 0.999 (p 1.0: everything 1.000, not drawn).
- (c) torus: path 1L 1.000; index 1L 0.719 (0.688-0.739), 4L 0.860 (0.814-0.881), looped x4 0.864 (0.860-0.868); floors
  blank 0.509, retrace 0.844.

**Caption (ready to paste; registered results, floors post hoc).** Accuracy of index-RoPE attention against its number of
layers (orange; dots: runs, line: means) beside one layer of path integration (blue line), each trained and tested at the
same length and depth. (a) Dyck-2 at length 32 and depth 12 (Hewitt closing accuracy A2f; n = 8): index attention needs
about three layers to approach one path layer (0.964 at 3, 0.982 at 4, vs 0.996). (b) Ring of 32 cells at T = 128 for
three move biases (n = 6 fresh seeds): three index layers reach one path layer (0.998-0.999 vs 1.000). (c) Torus at
T = 1024 (n = 4): four index layers (0.860) or one layer looped four times (0.864) stay far below one path layer (1.000),
just above the retrace floor (0.844). Dashed lines: map-free floors (Dyck: best A2f floor; ring: retrace per bias, on
another stream of the same task; torus: always-blank and retrace on this batch's own walks).

## F3: SOLVED by per-head rank (`fig_rank_solved.py`)

Data (parsed from the registered analysis outputs): `RANK_SEP_ANALYSIS.txt` (A, B, C_bd, C, D), `RANK3_ANALYSIS.txt` (E),
`RANK_ND_ANALYSIS.txt` (A2, B2, A3, B3), `RANK_WRAP_ANALYSIS.txt` (3L), `RANK_TESTS_ANALYSIS.txt` (R1, R2). The exact lines
drawn are copied into `fig_rank_solved_out.txt`.

Numbers: (a) torus T = 1024, n = 8: per-head rank 2 -- shared r 2 0/8 (acc 0.894), per-head r 2 2/8 (0.885), block-diagonal
r 4 2/8 (0.948); rank 3 -- per-head r 3 6/8 (0.987); rank 4 -- shared r 4 8/8 (0.998), per-head r 4 8/8 (0.999). (b) rank =
D vs D + 1: 1D ring 7/8 vs 8/8; 2D torus (32/side) 1/8 vs 8/8; 3D torus (10/side) 1/8 vs 4/8, and rank 4 on an 18/side 3D
torus 8/8.

**Caption (ready to paste; registered).** Runs (of 8) that solve the map, by the per-head rank of the content-to-angle map.
(a) Torus, T = 1024, six bottlenecks built from the same initial weights: every arm whose heads each read a rank-2 map
solves 0-2/8, rank 3 solves 6/8 and rank 4 8/8, whether the latent is shared or per head (RANK_SEP, RANK3). (b) Rank equal
to the task's dimension D against D + 1 (separate batches and recipes): on the 1D ring rank 1 is not hard (7/8 vs 8/8;
RANK_TESTS); on a 2D torus rank 2 is (1/8 vs 8/8; RANK_ND); on a small 3D torus (10 per side) both rank 3 and rank 4 are
(1/8, 4/8), and rank 4 solves 8/8 on an 18-per-side torus with few wrap-only revisits (hatched; RANK_WRAP).
Budget-scoped (900 epochs on the torus; 2x the budget does not rescue rank 2: LOOP_RANK_E1800_P1).

## F5: existence before mechanism (`fig_existence.py`)

Data (parsed from registered outputs): `RANK_SEP_ANALYSIS.txt` (A, C from scratch), `RANK_PROJ_ANALYSIS.txt` (FROZEN mean;
TRAINABLE r2-proj, CONTROL r4-cont, REFERENCE r2-cont class counts; lr restart 1e-3), `WRAP_EXISTENCE_ANALYSIS.txt` (E2
PRIMARY, E3), `GENERALITY_ANALYSIS.txt` (E2 PRIMARY at lr 1e-4, E3).

Numbers (`fig_existence_out.txt`): (a) rank 2 torus: from scratch 0/8; r 2 continued 1/8; projected r 2 installed (frozen
0.9955) and trained 7/8; r 4 from scratch 8/8; r 4 continued 7/8. (b) ring |.|: from scratch 0/8 (acc 0.921); installed
(frozen = source, 1.000) and trained 8/8, lost 0/8 (HELD); signed 8/8 / 8/8, lost 0/8. (c) torus |.|: from scratch 0/8
(0.880); installed and trained 8/8 but lost 6/8 (peak loss 0.11-0.21, re-found by epoch 119-204: NOT HELD, re-found);
signed 8/8 / 8/8, lost 0/8.

**Caption (ready to paste; registered).** A solution exists in each constrained class and training keeps or re-finds it,
but training from scratch does not find it. Bars: runs SOLVED of 8 (hatched: the solution was installed and then trained).
(a) Rank 2 on the torus: a rank-2 projection of a solved rank-4 model scores 0.9955 frozen and is held when trained (7/8),
while rank 2 from scratch solves 0/8, and 1/8 when the stalled runs are trained further (RANK_SEP, RANK_PROJ). (b) The |.|
step on the ring: an installed exact non-negative solution is held under training (8/8 SOLVED, lost on 0/8), while |.|
from scratch solves 0/8 against signed 8/8 (WRAP_EXISTENCE). (c) The |.| step on the torus: the installed solution is
perturbed (loss above 0.05 on 6/8) and re-found (8/8 SOLVED); from scratch 0/8 vs 8/8 (GENERALITY). "Lost" (registered
definition): the max per-epoch training loss after the start exceeds 0.3, or exceeds 10x the signed control's max at that
seed and 0.05.

## F6: navigation told in words (`fig_textworld.py`)

Data: `TEXTWORLD_SECONDARY.json` only (declared secondaries S1, S3, S5 of the registered TEXTWORLD batch; path arm
Vanilla_r4, 1 layer, T = 1024, seeds 0-7, seeds 0-1 were the pilot). "Clock" seeds: |c| / |north| > 0.3 (s0, s2, s3, s4;
values 0.48-1.14 vs 0.04-0.06; cos(c, verb step) +0.999 to +1.000). Ablation accuracies are eval mode without the 1/(1-p)
attention re-scale (so s0 and s7 read 0.88; re-scaled path mean 0.993, TW_NORMSTEP CORRECTED block).

Numbers (`fig_textworld_out.txt`): opposition raw 0.080-0.120 (map seeds) and 1.43-1.77 (clock seeds); after removing the
common component 0.002-0.012 on all 8 seeds; ablation: all steps 0.88-1.000; non-direction steps zeroed 0.59-0.95 (map
seeds) and 0.19-0.67 (clock seeds); direction steps zeroed 0.39-0.51; floors constant 0.505, reversal-copy 0.597.

**Caption (ready to paste; registered batch, declared secondaries).** What a one-layer path model learns from a torus walk
told in English (TEXTWORLD, n = 8 seeds). (a) Opposition of the learned direction-word steps, |A + B| / |A| averaged over
north/south and west/east (0 = opposites cancel): after removing the component common to all four directions, opposites
cancel to 0.002-0.012 on every seed. (b) That common component is small on four seeds (blue, 4-6% of a direction step) and
large on the other four (orange, 0.48-1.14), where it points along the verbs' step: a per-step clock layered on the map
(31-38 of 64 phase channels drift between visits to the same cell). (c) Held-out accuracy with every step, with the steps
of all non-direction words zeroed, and with the direction words' steps zeroed: removing the direction steps drops every seed
to the constant floor (0.505, dashed) or below; the clock seeds also depend on the other words. Eval mode without the
dropout re-scale. Seeds 0-1 were the pilot.

## Skipped / not made (2026-10-16)
- Nothing in outline section 4 needed a GPU; every listed CPU figure (F1-F9) now exists. T1-T4 are tables (text), not made
  here.
- F4's ring half uses WRAP_EXISTENCE only (one ring batch, as the review asked); RING_DIAL / MECHANISM / ROBUST-ring loss
  curves exist in the same checkpoints if wanted.
- `fig_mechanism` and `fig_ring_strata` were fixed in place, not restyled to the shared palette.

## Draft-2 figures from the audit-corrected generators (2026-10-21; post hoc, CPU, eval-only)

Built for `docs/paper/MAIN_DRAFT2.md` from the generators that `docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md`
corrected; each script asserts its drawn means equal to the committed text outputs (3 decimals) before drawing.

### `fig_segments.py` (plan Fig. 8; draft-2 Fig. 4, Sec. 6)

Data: `expansion/scripts/depth/out/segments_bins.json`, written by the new `depth/segments_bins.py` from the per-revisit
rows of `torus_depth.py` / `ring_depth.py` (`$DEPTH_SCRATCH/*.npz`, not committed; regenerate in minutes). Seed means
(mean of per-seed accuracies; audit M-DP4), dots = seeds. (a) Torus T = 1024, FACTORIAL s500-503: solid = plain revisits
at gap < 128 (2 runs = retrace; = Table 3 / `torus_depth_out.txt`), dashed = plain revisits at gaps 32-127
(`torus_segments_out.txt`). (b) Ring T = 128, p_plus 0.5, CANCEL fresh seeds 2-7: 2 runs = retrace, 3+ = not retrace, not
a one-lap return, not wrap-only (audit depth MINOR; = `ring_plain_bins_out.txt` 'plain only'). Floors: retrace exact at 2
runs; best constant = torus short-gap stratum (`torus_strata_floors_s500-503.json`, 0.508) / ring 'neither' class
(`ring_class_floors_out.txt`, 0.493; includes wrap-only). 1L curves labelled eval mode (no dropout rescale).

| bin (runs) | 2 | 3-4 | 5-8 | 9-16 | 17+ |
|---|---|---|---|---|---|
| torus index 1L, gap < 128 | 0.833 | 0.477 | 0.476 | 0.474 | 0.472 |
| torus index 4L, gap < 128 / gap 32-127 | 0.996 / 0.874 | 0.926 / 0.832 | 0.735 / 0.688 | 0.561 / 0.560 | 0.493 / 0.493 |
| torus loop x4, gap < 128 / gap 32-127 | 1.000 / 0.996 | 0.977 / 0.911 | 0.645 / 0.575 | 0.499 / 0.499 | 0.502 / 0.502 |
| torus path 1L | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ring index 1L / 2L / 3L (plain) | 0.797 / 0.987 / 1.000 | 0.491 / 0.910 / 0.999 | 0.473 / 0.828 / 0.996 | 0.454 / 0.742 / 0.988 | -- |

### `fig_clock_endstate.py` (plan Fig. 9; draft-2 Fig. 5, Sec. 7.1)

Data: `expansion/scripts/sign/clock_table_out.txt` per-run run ratios (symmetric map, usage pooled over heads; B3 /
M-SG1; one definition for every row) and its SOLVED stars; collapsed = unsolved rank-2 run with both head ratios < 0.15
and COLLAPSE in `expansion/scripts/rank/sweep_final_out.txt` (per-head r2 s5, s6; block-diagonal s1, s7). Values are the
3-decimal printed ones (torus signed small start 0.001-0.002). Rows: signed small start (MECH SO, SO_absmag, GAUGE SO_m,
TORUS_START SO4) 0.001-0.036, 31/32 SOLVED; rank 4 0.010-0.136; rank 3 solved 0.017-0.081, unsolved 0.767-1.105; rank 2
solved 0.045-0.611; rank 2 unsolved 0.053-1.263 (median 0.624; collapsed 0.053-0.086); signed frozen latents unsolved
1.055-1.741, solved 0.058; |.| ring small start 0.698-1.850 (solved 0.881); |.| torus unsolved 0.700-1.868 (solved 0.024);
|.| past pi ring 0.106-0.757 (8/8); signed large start (SO_high, SO4_high) 0.004-1.777, 10/16, the solved torus runs
1.390 / 1.777 with near-pi usage 0.43 / 0.99. No side panel: RANK's off-plane share is a different (latent) metric (audit
M-RK1 / M-RK2).

### `fig_kernel_peak.py` (plan Fig. 11; draft-2 Fig. 10, Sec. 7.4)

(a) `expansion/scripts/mappope/kernel_argmax_odd_out.txt`: argmax of K_ao(D) over odd D (audit mappope MINOR), PAPER2X2
index arms, 1,088 pooled pairs: PoPE-Flat D = 1 0.982, 7 0.015, 11 0.004; RoPE 3 0.523, 7 0.314, 11 0.143, 15 0.019;
K(3) > K(1) per run RoPE 0.53-1.00, PoPE 0.00 (`kernel_probe_paper2x2_out.txt`). Drawn as the argmax distribution; no
committed output holds the kernel curve K(D) itself. (b) `kernel_probe_paper2x2.json`, stratum 'all', per seed: MapWM C
0.703 vs C0 0.905 (C < C0 on 7/8), MapPoPE-Flat 0.995 / 0.996. (c) `kernel_probe_score_rank_hstratum.json` (HEAD_RULE=
stratum, audit B2), C0 means: rank 2 RoPE score 0.900 / 0.795 / 0.399, polar 0.960 / 0.905 / 0.482; rank 4 RoPE score
1.000 / 0.993 / 0.967, polar 1.000 / 0.993 / 0.970 (gap < 128 / >= 128 / wrap-only; = `probe_stats_hstratum_out.txt`).
