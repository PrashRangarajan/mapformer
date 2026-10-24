# Does the sign cost appear in a state-space model? -- results (G-3, registered, `G3_SSM_PREREG.md` + Amendment 1)

Batch 2026-10-23 18:24 -> 23:02 on koopman GPU 1 (up to 4 runs at a time), configuration `minimal_h2` (launch-time choice,
prereg Sec. 10), 104 runs, code `ffa34cd` (the 104 md5s in `runs/ssm_sign/p0/code_md5.txt` match that commit; prereg md5
243a2bbe...), torch 2.10.0+cu128. **Model:** a one-layer diagonal complex SSM with an input-dependent rotation and no
decay, H_t = diag(exp(-i omega Delta(x_t))) H_{t-1} + k_t v_t^T, y_t = Re(conj(q_t)^T H_t), computed as unnormalised
rotary linear attention (Mamba-3 Prop. 3; equal to the step-by-step recurrence to 1.1e-12 in float64). Only the sequence
mixer differs from the attention ring arms: softmax with attention dropout is replaced by the causal linear kernel plus a
per-head RMS norm; every shared tensor starts bitwise equal to the attention arm's at the same seed (24/24). **Task:** the
attention ring anchors' (retrieval at revisits on Z_32, moves +1/-1, map redrawn per walk, only the two moves step),
T = 128, 300 epochs, lr 3e-3 cosine (5% warm-up), AdamW wd 0.05, rank 4, 1 layer, 2 heads, d 128. **Arms (block A):**
signed, |.| (abs), softplus, exp, index (every token steps 1: the integration gate). **Blocks H / H2 (rule 14):** each
solved signed source of block A converted exactly into |.|, softplus and exp form plus a signed control, evaluated frozen,
then trained 100 epochs on a fresh stream at lr 3e-3 (H) and 1e-3 (H2). Fresh seeds 2100-2107; pilots s2180-2189 (prereg
Sec. 9) not pooled. Independent results-blind check before launch: Amendment 1 (no BLOCKER, 6 MAJOR, 11 MINOR, all folded
in). Analysis `analyze_ssm_sign.py --config minimal_h2` -> `G3_SSM_ANALYSIS.txt`, `G3_SSM_VERDICTS.json`; re-run here from
the rsynced run dirs, the verdicts JSON is equal and the text identical. **Floors** (gate, registered evaluation stream):
blank 0.496, best n-gram 0.554, retrace 0.864 at T = 128 (0.498 / 0.544 / 0.809 at T = 512); memoryless pilot 0.496.

**Reading (registered): SIGN COST IN THE SSM.** Worded per Amendment 1.5: the learnability cost of a non-negative phase
step is **not specific to softmax attention** -- it appears in unnormalised rotary linear attention, which is this
one-layer diagonal complex SSM. From scratch at lr 3e-3, |.| solves 0/8 against signed 8/8 (Fisher p 0.0002, permutation
p 0.0002, accuracy 0.821 against 1.000, d -0.179); softplus solves 1/8 (SIGNED LEARNS BETTER on both tests, Holm). The
rival registered in Sec. 8, "softmax competition is the source of the cost", is not supported. m = 8 was not run in this
configuration.

## Block A (registered primary; lr 3e-3, m = 32, n = 8 per arm)

| arm | SOLVED | classes (S / T / D) | acc T = 128 | acc T = 512 (4x, descriptive) | RUNAWAY |
|---|---|---|---|---|---|
| signed | **8/8** | 8 / 0 / 0 | **1.000** +/- 0.000 | 0.999 +/- 0.002 | 0 |
| softplus | 1/8 | 1 / 2 / **5** | 0.935 +/- 0.061 | 0.617 +/- 0.164 | 0 |
| \|.\| (abs) | **0/8** | 0 / 7 / 1 | 0.821 +/- 0.075 | 0.457 +/- 0.044 | 0 |
| index (gate GI) | 0/8 | 0 / 5 / 3 | 0.755 +/- 0.024 | 0.445 +/- 0.048 | 0 |
| exp | 0/8 | 0 / 8 / 0 | 0.473 +/- 0.190 | 0.421 +/- 0.159 | **8** |

Mean +/- sd over seeds; none diverged. G0 passes (signed 8/8 >= 6); GI passes (index 0/8 <= 1, signed - index +0.245 >=
0.05). |.| from scratch (0.821) sits below the retrace floor (0.864) and near the index arm; at T = 512 it is at the index
arm's level (0.457 vs 0.445; retrace 0.809).

## Registered decisions

- **P (primary, abs vs signed, m = 32): SIGNED LEARNS BETTER** (gated label = raw label). Fisher p 0.0002 on SOLVED (0/8 vs
  8/8), permutation p 0.0002 on accuracy, d -0.179.
- **BR softplus vs signed: SIGNED LEARNS BETTER**, both tests survive Holm (Fisher 0.0014 / Holm 0.0014; permutation 0.0002 /
  Holm 0.0003; d -0.065). **Budget-scoped:** 5 of the 7 unsolved softplus runs were still DESCENDING at 300 epochs.
- **BR exp vs signed: SIGNED LEARNS BETTER (8 RUNAWAY)** (Fisher 0.0002 / Holm 0.0003; permutation 0.0002 / Holm 0.0003; d
  -0.527). **This is instability at that rate, not a learnability result.** All 8 exp runs are RUNAWAY (Amendment 1.4): 7
  by final move-step magnitude above 10^3 (up to 3.6e24), the eighth (s2103, step 689) by a post-warm-up training loss of
  2.17 above the memoryless 2.08; mean accuracy 0.473 is below the blank floor. Under Amendment 1.4 the label is kept with
  its suffix, licenses no learnability sentence and forms no joint; under prereg Sec. 8 an exp failure at 3e-3 reads as
  instability unless block A2 (1e-3) agrees, and A2 was not run. The attention anchors at 1e-3 (`SIGNROUTE_RESULTS.md`,
  ring, never pooled): exp 1/8 (0.935), softplus 1/8 (0.879), |.| 1/8 (0.917, LRSWEEP), signed 8/8; at 3e-3 exp 1/8
  (0.958), softplus 3/8 (0.966), |.| 2/8.

## Rule 14: the non-negative solution exists in the SSM (blocks H, H2)

**E1 (exact construction): passes for every class and the control.** On all 8 sources frozen accuracy equals the source's
(0.9997-1.0000), move-angle error <= 2.3e-6 rad, maximum logit difference <= 8.3e-3 at logit scale ~24 (float32 cos / sin
of larger cumulative angles). Weight scale (balanced, rule 14): max |W_in| = max |W_out| 1.20-1.74 for |.| against the
sources' 0.47-0.53 (<= 3.7x), <= 1.5x for softplus, <= 1.2x for exp. **Declared (rule 15):** constructions (f) and (g) fix
omega' at 2 pi / ln 2 = 9.06 and 2 pi against the sources' max |omega| 0.59-0.84; |.| uses omega' = max(|omega|, 1).
This is the first build of construction (g) (exp) in a trained model. Move-angle sensitivity (median over sources):
|.| 108, softplus 180, exp 224, signed control 27, source 21 (attention: AO_bal 87-96, SO_bal 15-18).

**E2 continuations** (100 epochs, a third of WRAP_EXISTENCE's recovery budget; LOST = max per-epoch loss > 0.3, or > 10x
the control's at that seed and > 0.05):

| block (lr) | class | SOLVED | LOST (by > 0.3) | max loss | acc T = 128 (control) | E2 (registered) | E2' (declared secondary, LOST = > 0.3 only) |
|---|---|---|---|---|---|---|---|
| H (3e-3) | signed control | 8/8 | -- | 0.006-0.039 | 1.000 | -- | -- |
| H (3e-3) | \|.\| | 8/8 | 8 (8) | 0.320-0.678 | 0.9999 (1.000) | **PERTURBED AND RE-FOUND** | PERTURBED AND RE-FOUND |
| H (3e-3) | softplus (f) | 3/8 (5 D) | 8 (8) | 1.215-1.481 | 0.974 (1.000) | NOT HELD [construction (f)] | NOT HELD |
| H (3e-3) | exp (g) | 1/8 (7 D) | 8 (8) | 1.491-1.811 | 0.930 (1.000) | NOT HELD [construction (g)] | NOT HELD |
| H2 (1e-3) | signed control | 8/8 | -- | 0.001-0.002 | 1.000 | -- | -- |
| H2 (1e-3) | \|.\| | 8/8 | 6 (0) | 0.033-0.127 | 1.000 (1.000) | **PERTURBED AND RE-FOUND** | **HELD** |
| H2 (1e-3) | softplus (f) | 8/8 | 8 (0) | 0.050-0.204 | 0.9999 (1.000) | PERTURBED AND RE-FOUND [construction (f)] | HELD |
| H2 (1e-3) | exp (g) | 8/8 | 8 (0) | 0.110-0.165 | 1.000 (1.000) | PERTURBED AND RE-FOUND [construction (g)] | HELD |

Attention continuation max-loss ranges on record (WRAP_EXISTENCE ring, lr 1e-3): AO_bal 0.024-0.154, SO_bal 0.015-0.112.

- **|.| (the class-level statement, Amendment 1.2):** at lr 3e-3 the installed |.| solution is **perturbed and re-found,
  not held**: every seed's loss peaks above 0.3 (at epoch 4-26) and every seed ends SOLVED (0.9999). At lr 1e-3
  it ends SOLVED on 8/8 with no seed above 0.3 (peaks 0.033-0.127, inside attention's held range); the registered E2 reads
  PERTURBED AND RE-FOUND only through the 10x-control clause (the SSM signed control peaks at 0.001-0.002), which
  Amendment 1.3 anticipated, and **the declared secondary E2' reads HELD**.
- **softplus / exp (Amendment 1.2):** every non-HELD label is read as a property of constructions (f) / (g) at
  omega' = 2 pi / ln 2 and 2 pi, not of the class. At 3e-3 most continuations were still DESCENDING at 100 epochs, so the
  line between NOT HELD and PERTURBED AND RE-FOUND is set partly by the budget (prereg Sec. 4, A1.7).

**Joint (registered):** |.| at 3e-3: **COST, SOLUTION PERTURBED AND RE-FOUND**; softplus at 3e-3: COST, SOLUTION NOT HELD
(construction (f)). H2 joins block A's 3e-3 label as a declared CROSS-RATE joint (from scratch at 3e-3, kept at 1e-3): |.|
COST, SOLUTION PERTURBED AND RE-FOUND; softplus the same (construction (f)). exp forms no joint (RUNAWAY). **The registered
SEARCH COST joint (which needs HELD) did not form at either rate.** On E2' (secondary, cross-rate) |.| reads HELD at 1e-3,
i.e. exists, is kept and is not found from scratch -- a secondary reading only.

## Checks on the reading

- **Rule 2:** r(log10 final loss, acc) = -0.674 at T = 128 and -0.927 at T = 512 over the 40 block-A runs (Amendment 1.7:
  continuations excluded). The primary does not rest on it: signed is at 1.000 on every seed and the contrast fires on
  SOLVED (0/8 vs 8/8) as well as on accuracy.
- **Rule 3 / convergence:** `python3 -m mapformer.experiment_audit --runs-dir runs/ssm_sign/p0/{A,H,H2}` (exit 2: no
  (json, pt) pairs in its layout, so it runs check [1] only). Its windowed classes agree with the registered ones in every
  cell: A signed 8 SOLVED, abs 7 STALLED + 1 DESCENDING, softplus 1 + 2 + 5, exp 8 STALLED, index 5 STALLED + 3
  DESCENDING; H abs 8 SOLVED, softplus 3 SOLVED + 5 DESCENDING, exp 1 + 7 DESCENDING; H2 all 32 SOLVED. Its "not all arms
  converged" flag concerns softplus, exp and index in A and softplus, exp in H; the primary arms are converged (signed) or
  stalled (|.|, 7 of 8 flat).
- **Power (prereg Sec. 7):** with signed 8/8, SOLVED fires for |.| at <= 3/8; |.| at 0/8 is the strongest reachable outcome.

## Mechanism (descriptive, registered as such)

Medians per arm from the trained models (u-ratio = |z_+ + z_-| / (|z_+| + |z_-|) of the move latents; cancel share = share
of non-blind channels whose move-angle sum is within 0.1 rad of 0 mod 2 pi; "by wrap" = of those, |sum| > 1 rad):

| block | arm | u-ratio | cos(z_+, z_-) | cancel share (by wrap) |
|---|---|---|---|---|
| A | signed | 0.001 | -1.000 | 1.00 (0.00) |
| A | \|.\| | 0.592 | -0.334 | 0.07 (0.03) |
| A | softplus | 0.933 | +0.738 | 0.12 (0.11) |
| A | exp (RUNAWAY) | 0.943 | -1.000 | 0.05 (0.00) |
| H / H2 | \|.\| continuation | 0.752 / 0.715 | +0.129 / +0.023 | 0.28 / 0.43 (all by wrap) |
| H / H2 | signed control | 0.007 / 0.153 | -1.000 / -0.953 | 1.00 / 0.99 (0.00) |

The signed SSM ends where signed attention ends (antipodal move latents, every non-blind channel cancelling through zero;
SIGNROUTE at 3e-3: u 0.001, cos -1.000), and |.| never approaches it (SIGNROUTE |.| at 3e-3: u 0.664, cos -0.165). The
held / re-found |.| continuations cancel only by wrapping; their cancel shares are unweighted (every channel equal), which
understates the code in the channels attention reads (ACCOUNTS T1). Descriptive at T = 512: |.| continuations 0.953 (H)
and 0.971 (H2) against the controls' 0.999.

## Per seed (accuracy at T = 128 and class: S SOLVED, T STALLED, D DESCENDING; ! = RUNAWAY)

```
blk  arm       2100    2101    2102    2103    2104    2105    2106    2107
A    signed    1.000S  1.000S  1.000S  1.000S  1.000S  1.000S  1.000S  1.000S
A    abs       0.777T  0.869T  0.750T  0.761T  0.743T  0.825T  0.939T  0.903D
A    softplus  0.963D  0.920D  0.986D  0.898T  0.993S  0.817T  0.989D  0.912D
A    exp       0.568T! 0.496T! 0.521T! 0.681T! 0.031T! 0.497T! 0.496T! 0.496T!
A    index     0.774D  0.782T  0.775D  0.705T  0.751D  0.750T  0.749T  0.753T
H    abs       1.000S  1.000S  1.000S  1.000S  1.000S  1.000S  1.000S  1.000S
H    softplus  0.992S  0.988D  0.927D  0.989D  0.945D  0.953D  0.997S  0.999S
H    exp       0.946D  0.951D  0.877D  0.998S  0.978D  0.855D  0.956D  0.876D
H2   abs/softplus/exp/signed: 1.000S on every seed (softplus 0.9999 mean)
```

## Notes against the pre-registration

- **Host.** Prereg Sec. 5 names kalman; the batch ran on koopman (the second machine, same torch 2.10.0+cu128, md5 guard
  over the committed tree). Wall time 4 h 38 min on one GPU, below the Sec. 10 estimate (6.4-7.8 h, measured on a shared
  kalman).
- **Seed numbers.** Prereg Sec. 5 calls 2100-2107 fresh; RANKWD (torus attention, registered the same day, run on koopman)
  used the same seed numbers. No run, checkpoint or contrast is shared between the two batches; the overlap is in seed
  labels only.
- **Predictions (Sec. 8).** SIGN COST at m = 32 and BR SIGNED LEARNS BETTER for softplus and exp came out as predicted, E1
  passed for every class, and abs read PERTURBED AND RE-FOUND at 3e-3 as the pilot suggested. "HELD for abs at 1e-3" did not
  hold under the registered E2 (10x clause); it holds under the declared secondary E2'.

## Reading for the paper

The sign cost is not a property of softmax attention. In a one-layer diagonal complex SSM with an input-dependent
rotation -- the same function as unnormalised rotary linear attention -- a non-negative step learns the ring map on 0/8
seeds against 8/8 for a signed step, the trained signed SSM ends in the same antipodal, through-zero cancelling state as
signed attention, and an exact non-negative solution exists in the SSM (frozen accuracy = source on 8/8) and is re-found
after a perturbation when installed (8/8 SOLVED at both rates; held by the E2' criterion at 1e-3). Softplus also learns
worse (1/8, budget-scoped). Representability of rotations with non-negative angles in SSMs is Grazzi et al.'s and
Mamba-3's; that the non-negative rotary step is not learned there is ours.

## Scope

One layer, one task (ring of 32 cells, moves only stepping), one rate from scratch (3e-3; no SSM rate sweep), 300 epochs,
no decay, no output gate, Delta entering only the phase (not Mamba-faithful, declared). exp's 0/8 is instability at 3e-3
(8/8 RUNAWAY), not a learnability statement; softplus's 1/8 is budget-scoped (5 runs DESCENDING); m = 8, the second rate
from scratch (A2) and the every-token-steps block (B) were not run in this configuration. The registered SEARCH COST
joint did not form: at 3e-3 the |.| solution is perturbed and re-found, not held; at 1e-3 it is held only by E2'.
