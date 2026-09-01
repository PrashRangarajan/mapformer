# What is Level 1.5 actually made of? -- the decomposition at n=5

Clean torus paper task, 6 arms x 5 seeds, 300 epochs warmup+cosine, held-out map.
This is a RETRAIN, not a comparison to the published single-seed table (those used
16 epochs of LinearLR-from-step-one; rule 10).

The published decomposition rested on **one seed per arm**. At n=5, most of it
does not survive, and the one contrast that clears its noise floor turns out to be
a convergence gap rather than an architectural one.

## Raw accuracy (mean +/- sd over 5 seeds)

| arm | what it keeps | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| Vanilla | nothing (no filter at all) | 0.966 +/- 0.057 | 0.891 +/- 0.067 | 0.768 +/- 0.090 |
| Level15 | wrap + measurement + per-token R | 0.979 +/- 0.046 | 0.948 +/- 0.082 | 0.888 +/- 0.123 |
| L15_DARE | same, Pi fixed by DARE | 0.989 +/- 0.024 | 0.940 +/- 0.064 | 0.843 +/- 0.103 |
| L15_ConstR | wrap + measurement, NO per-token gate | 0.989 +/- 0.015 | 0.948 +/- 0.047 | 0.889 +/- 0.064 |
| L15_NoMeas | the wrap alone (z == 0) | 0.964 +/- 0.072 | 0.875 +/- 0.185 | 0.815 +/- 0.184 |
| L15_NoCorr | correction zeroed (== vanilla) | 0.936 +/- 0.043 | 0.862 +/- 0.052 | 0.748 +/- 0.055 |

## Paired contrasts, raw

Positive = first arm higher. DETECTABLE means |mean| > MDE = 2.8*sd/sqrt(5).

| contrast | T=128 | T=512 | T=1024 |
|---|---|---|---|
| Level15 - L15_NoMeas | +0.015 (MDE 0.033) | +0.074 (MDE 0.137) | +0.073 (MDE 0.105) |
| Level15 - L15_ConstR | -0.010 (MDE 0.068) | +0.000 (MDE 0.140) | -0.002 (MDE 0.206) |
| Level15 - L15_DARE | -0.010 (MDE 0.071) | +0.008 (MDE 0.132) | +0.045 (MDE 0.209) |
| Level15 - Vanilla | +0.014 (MDE 0.104) | +0.057 (MDE 0.150) | +0.120 (MDE 0.180) |
| **L15_ConstR - L15_NoCorr** | **+0.053** (MDE 0.049) | **+0.086** (MDE 0.046) | **+0.141** (MDE 0.050) |
| L15_NoCorr - Vanilla | -0.029 (MDE 0.051) | -0.029 (MDE 0.090) | -0.020 (MDE 0.112) |

**Exactly one contrast clears its MDE**, and it is the one nobody predicted:
ConstR > NoCorr, 5/5 seeds, t = 3.0 / 5.2 / 7.9. Everything else -- including the
headline Level15 - Vanilla -- is UNMEASURED at n=5.

## The convergence confound, and what survives it

Convergence is heterogeneous, so rule 9 applies before any of the above is read as
architecture. r(final training loss, accuracy) = **-0.930 / -0.897 / -0.812** over
all 30 runs at T=128/512/1024. Final loss by arm (|slope| < 5e-4 over the last 10%
of epochs = converged):

| arm | mean final loss | converged | per-seed |
|---|---|---|---|
| L15_DARE | 0.0569 | 4/5 | 0.2671 0.0002 0.0083 0.0001 0.0086 |
| L15_ConstR | 0.0593 | 3/5 | 0.0337 0.0173 0.0023 0.1182 0.1251 |
| Vanilla | 0.0760 | 3/5 | 0.0087 0.2676 0.0164 0.0255 0.0616 |
| Level15 | 0.0841 | 4/5 | 0.0002 0.0001 **0.4198** 0.0001 0.0002 |
| L15_NoMeas | 0.1546 | 1/5 | 0.1018 0.0010 **0.6260** 0.0164 0.0278 |
| L15_NoCorr | 0.1952 | 2/5 | 0.1778 0.1929 0.0629 0.0560 0.4866 |

NoCorr is the WORST-CONVERGING arm in the set. So the one raw-detectable contrast
is exactly the contrast against the arm that trained worst. Regressing accuracy on
final loss and comparing residuals:

| contrast, loss-matched | T=128 | T=512 | T=1024 |
|---|---|---|---|
| L15_ConstR - L15_NoCorr | +0.016 (t 1.24) | +0.014 (t 0.51) | +0.062 (t 1.56) |
| **Level15 - Vanilla** | +0.016 (t 1.32) | **+0.062 (t 3.08)** | **+0.124 (t 3.83)** |
| Level15 - L15_NoMeas | -0.004 (t -1.20) | +0.036 (t 1.23) | +0.031 (t 1.39) |

The two readings SWAP. ConstR - NoCorr stops being detectable; Level15 - Vanilla
starts being detectable, at T=512 and T=1024 only.

## Verdict against the pre-registration

- **Branch C fires, with the sign inverted.** "Removing the per-token gate is WORSE
  than doing nothing" (n=1: ConstR 0.672 < NoCorr 0.833) is REFUTED. At n=5 ConstR
  is one of the two best arms, 5/5 seeds above NoCorr. **Retract that claim.**
- **Branch A does not fire.** "NoMeas << Level15", the basis for "Level15 does not
  reduce to clamping theta", is UNMEASURED at n=5 raw (+0.074, MDE 0.137) and
  loss-matched (+0.036, t 1.23). The n=1 gap (0.831 vs 0.993) does not replicate.
  This does not establish B either -- it is unmeasured in both directions.
- **Branch D does not fire**, but only after loss-matching: raw, Level15 - Vanilla
  is inside its MDE at every length. NoCorr == Vanilla holds as designed
  (-0.029/-0.029/-0.020, all unmeasured), so the ablation harness is sound.
- **L15_DARE == Level15** at every length. The principled DARE gain buys nothing a
  learned scalar does not; consistent with the n=1 read, still underpowered.

**No single piece can be shown to be load-bearing.** Remove the measurement head,
remove the per-token gate, or replace the learned Pi with the DARE solution -- each
one alone costs nothing measurable. Only removing the correction ENTIRELY costs
anything, and that only shows up at OOD length after loss-matching. The filter's
effect is real (Level15 - Vanilla, loss-matched, t 3.08 / 3.83) and is confined to
OOD length -- which is the STABILISATION signature the project already reported,
now with the added result that the decomposition into named parts is not supported.

## Scope

Clean config only. The lm200 column of the original table is under the 2026-07-16
convergence retraction and is not reproduced here. n=5, one task, one width. The
loss-matched analysis is a regression control, not a randomised one; the honest
summary is that this design cannot separate the arms, not that the arms are equal.
