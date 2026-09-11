# D5 at n=24: magnitude freedom buys NOTHING, and the headline effect was overestimated 1.9x

Pre-registration: `D5_PREREG.md`. `runs/dof/recency` extended with seeds 8-23;
4 arms x 24 seeds.

| arm | phase DOF | final loss | acc T=1024 |
|---|---|---|---|
| `AlignFree` | `n_b` | **0.746 +/- 0.362** | **0.818 +/- 0.127** |
| `sep` (random) | `n_b` | 0.775 +/- 0.299 | 0.814 +/- 0.106 |
| `P0` (single) | 0 | 1.143 +/- 0.320 | 0.687 +/- 0.127 |
| `AlignLock` | 0 | 1.239 +/- 0.388 | 0.654 +/- 0.141 |

## E1 and E2 REFUTED -- and E4 is why we know it is noise, not power

| readout | n=8 | seeds 8-23 only (n=16) | pooled n=24 | verdict |
|---|---|---|---|---|
| accuracy `AlignLock - P0` | **+0.120** (6/8) | **-0.109** (3/16) | -0.033 (9/24, MDE 0.116) | **unmeasured** |
| loss `AlignLock - P0` | **-0.292** (2/8) | **+0.290** (13/16) | +0.096 (15/24, MDE 0.310) | **unmeasured** |

**E4 fired exactly as written.** Its falsifier was "the new seeds alone give a
different sign, in which case the n=8 result was noise and the pooled number should
not be quoted". Both readouts flip sign on fresh seeds. The n=8 estimates were
noise, and the pooled n=24 values are reported only to show they sit at zero.

**Magnitude freedom buys nothing.** The per-block scale ratio is free in
`AlignLock`, is exercised (1.0 -> 0.794), never goes negative, and changes neither
fit nor accuracy.

## E5 held, and phase freedom is now the WHOLE effect

| contrast at n=24 | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| `sep - P0` (total) | **+0.128** | 0.190 | 0.108 | 17/24 | **DETECTABLE** |
| `AlignFree - AlignLock` (phase) | **+0.165** | 0.155 | 0.088 | **21/24** | **DETECTABLE** |
| `AlignLock - P0` (magnitude) | -0.033 | 0.203 | 0.116 | 9/24 | unmeasured |
| `sep - AlignFree` (init coherence) | -0.004 | 0.133 | 0.076 | 10/24 | unmeasured |

Phase freedom (+0.165) is now LARGER than the total effect it decomposes (+0.128),
because the other two components are slightly negative. At n=8 the decomposition
read +0.148 / +0.120 / -0.031 and looked like two live components; at n=24 it is
**one component and two zeros**. The pre-registration's stated limit -- that a
single ladder cannot attribute phase versus magnitude -- turned out not to bind,
because magnitude simply has no effect to attribute.

## E3 CONFIRMED as registered

`r(final loss, acc) = -0.986` over 96 runs. Loss-matched residuals:
`AlignLock - P0` **+0.001** (MDE 0.021), `AlignFree - AlignLock` **-0.009**
(MDE 0.015). Both zero, as E3 predicted in advance. On the clock task the
origin-vector effect remains entirely an effect on FIT.

## The correction that matters most: D4's "exact reproduction" proved the wrong thing

`DOF_RESULTS.md` reported `sep - P0 = +0.237` against a published **+0.237** and
called it reproduction to three decimals. **At n=24 the same contrast is +0.128** --
the n=8 estimate was high by a factor of 1.9.

The two batches that "agreed" used the **same seeds 0-7**. They therefore measured
pipeline determinism, which is real and worth knowing, and NOT the stability of the
effect size, which is what I read into it. A same-seed rerun cannot validate an
effect size no matter how many decimals it matches. Rule 6 says three seeds is not
a point estimate; this says eight is not one either, and that an agreement between
two same-seed samples is one sample.

Every n=8 number in `DOF_RESULTS.md` and `RECENCY_EM_RESULTS.md` inherits this.
The DIRECTIONS all survive the extension -- `sep > P0`, phase freedom positive,
coherence null -- and D1 got *stronger* (+0.148 -> +0.165, 8/8 -> 21/24). The
MAGNITUDES should be quoted from this file.

## Standing

- **Phase freedom is the mechanism**, at n=24, 21/24 seeds, detectable.
- **On the clock task it acts on fit, not representation** (rule 9, twice now).
- **Magnitude freedom and initial coherence are both null** at MDE ~0.08-0.12.
- The map-side half of `DOF_RESULTS.md` (freedom COSTS 0.088-0.154 at OOD length,
  at `r(loss,acc) = -0.16`) is still n=8 and has NOT been extended. Given what
  happened here, its effect size should be treated as provisional until it is.
