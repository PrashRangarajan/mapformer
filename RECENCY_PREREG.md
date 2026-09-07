# Recency (k-back): pre-registration

**Status: pilot run FIRST. This is not a blind pre-registration and is not
labelled as one.** A 12-epoch CPU smoke test of three arms was run to check the
trainer worked, and it showed the direction before this file was written. What
follows pre-registers everything the pilot does NOT settle, with the pilot stated
in full so nobody later mistakes it for the result.

## The pilot, in full

n=1 seed, 12 epochs, batch 8, T=128, CPU, `--n-batches 12`. Chance 0.0625;
most-recent shortcut floor 0.1797.

| arm | acc | nll | per-offset k1..k8 |
|---|---|---|---|
| `Signed_r4` | 0.424 | 1.990 | 0.92 0.57 0.42 0.38 0.19 0.35 0.29 0.28 |
| `Abs_r4` (monotone) | 0.667 | 1.233 | 0.92 0.76 0.67 0.67 0.64 0.53 0.54 0.63 |
| `RoPE` (index) | 0.782 | 1.012 | 0.88 0.71 0.84 0.75 0.66 0.76 0.88 0.81 |

**What this establishes:** the task is learnable, all three arms sit off both the
floor and the ceiling, and the pipeline runs end to end.

**What it does NOT establish: anything numeric.** n=1, a budget far below any
used for a result here, CPU, and a training length a quarter of the intended one.
Standing rules 5 and 6 apply with full force -- this project has had three
successive claims off n<=3 dissolve in a single session, and a 16-epoch budget
produce three false negatives in a single day.

## Pre-registered, and not settled by the pilot

**H1 (the crossover interaction) -- the primary claim.**
`(monotone - signed)` on recency, against the same contrast on the torus, where
it is measured at **-0.215 / -0.280** at T=512/1024 (12/12 seeds, matched loss).
Predicted **positive on recency**, giving a sign-reversing interaction. n=8 in one
batch, arms `Signed_r4` / `Abs_r4` / `Pos_r4` / `CARoPE_r4`, loss-matched per
rule 9 and with r(final loss, acc) reported per length before any residual is
leant on.

**H2 (the more interesting one) -- alpha as a diagnostic.** The unconstrained
`Signed_r4` arm is FREE to learn a monotone code. Predicted: on recency it learns
opposition high and alpha near 0.94, on the torus opposition 0.11 and alpha 0.52
-- i.e. the accumulator exponent measured on a trained model READS OFF what the
task demanded. Measured with `probe_sign.py` / `probe_accumulator.py`. This is
the claim that would make alpha predictive rather than descriptive, and the pilot
says nothing about it.

**H3 (mechanism readout).** Per-offset accuracy should be markedly steeper in k
for signed than monotone, because signed collisions grow with reach. Pre-register
the SLOPE, not the mean: a mean gap with matched slopes would mean the arm is
worse for some other reason. The pilot is consistent with this and is n=1.

**H4 (the full position axis inverts).** Index (`RoPE`, `PlainFlat`) should be at
or above the path-integrated arms here, against +0.461 behind them on the torus.
The pilot's RoPE is the best arm, which if it holds is a stronger statement than
the sign result alone.

## Falsifiers, committed before the batch

- **If `signed >= monotone` on recency at n=8 (loss-matched), the clock/map
  dichotomy is WITHDRAWN from `positional_review.tex` and `mapformer_math.tex`.**
  It would also explain why alpha covers only 2 of the 4 mechanisms it was
  introduced for.
- If every arm lands within the noise floor (0.150), the task is uninformative
  and no reading is taken in either direction -- rule 11, "unmeasured", not
  "null".
- If the per-offset slopes match while the means differ, H3 fails and the mean
  gap is attributed to something else, named before it is explained.

## Checks required before any number is read

1. **Convergence** (rule 10): loss slope over the final 10%, per arm, both
   schedules if any arm is flat. `--schedule cosine` is the default here.
2. **Budget sensitivity** (rule 5): the headline re-run at 2x epochs. A weak
   number at one budget is not a result, and two budget points are not a trend.
3. **MDE** (rule 11) reported beside every contrast, and "unmeasured" used where
   the effect is under it.
4. **One batch** (rule 3): every arm retrained together, never against a stored
   checkpoint.
5. Gates already pass at the default `min_gap = k_max` for T=256..2048
   (`RECENCY_GATES.md`); re-run them if any task parameter changes.

## The ceiling trap, explicitly

The sign ablation's pre-registered discriminator was unmeasurable by construction
-- it asked for a deficit at a training length where the baseline is
1.000 +/- 0.000. Checked here: the pilot's arms span 0.42-0.78 at T=128 with the
floor at 0.18, so every verdict cell can go either way. If the real budget pushes
any arm to ceiling, the primary contrast moves to a longer T where headroom
remains, and that substitution is recorded here rather than chosen afterwards.
