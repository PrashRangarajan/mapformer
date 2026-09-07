# Does the OOD benefit localise to under-trained channels?

Pre-registered in `LOCALISATION_PREREG.md`. Eval-only.

## P1 — are any channels under-trained? (`n_cycles < 1`)

| arm | T | mean range(S) | under-trained channels |
|---|---|---|---|
| `Abs_r4` | 1024 | 649.77 | 0.8 / 64 |
| `Abs_r4` | 128 | 91.47 | 17.0 / 64 |
| `Signed_r4` | 1024 | 269.65 | 4.3 / 64 |
| `Signed_r4` | 128 | 91.84 | 8.2 / 64 |

## P2 — ablating channels, accuracy change vs unablated

Negative = ablation hurts. **Predicted: `low` hurts LESS at T=1024 than at T=128, and less than `high`.**

| arm | T | k | low | high | rand |
|---|---|---|---|---|---|
| `Signed_r4` | 128 | 4 | +0.000 ± 0.000 | -0.002 ± 0.003 | +0.000 ± 0.000 |
| `Signed_r4` | 128 | 8 | +0.000 ± 0.000 | -0.060 ± 0.076 | -0.001 ± 0.002 |
| `Signed_r4` | 128 | 16 | +0.000 ± 0.000 | -0.208 ± 0.166 | -0.016 ± 0.017 |
| `Signed_r4` | 128 | 32 | -0.001 ± 0.001 | -0.500 ± 0.029 | -0.197 ± 0.112 |
| `Signed_r4` | 1024 | 4 | -0.004 ± 0.007 | +0.002 ± 0.021 | -0.003 ± 0.009 |
| `Signed_r4` | 1024 | 8 | -0.021 ± 0.011 | -0.071 ± 0.104 | -0.010 ± 0.022 |
| `Signed_r4` | 1024 | 16 | -0.028 ± 0.013 | -0.210 ± 0.133 | -0.031 ± 0.020 |
| `Signed_r4` | 1024 | 32 | -0.139 ± 0.092 | -0.415 ± 0.018 | -0.256 ± 0.053 |
| `Abs_r4` | 128 | 4 | -0.068 ± 0.086 | -0.239 ± 0.242 | -0.128 ± 0.091 |
| `Abs_r4` | 128 | 8 | -0.187 ± 0.160 | -0.514 ± 0.275 | -0.209 ± 0.181 |
| `Abs_r4` | 128 | 16 | -0.254 ± 0.182 | -0.683 ± 0.181 | -0.509 ± 0.249 |
| `Abs_r4` | 128 | 32 | -0.280 ± 0.206 | -0.688 ± 0.204 | -0.597 ± 0.162 |
| `Abs_r4` | 1024 | 4 | -0.040 ± 0.033 | -0.061 ± 0.060 | -0.060 ± 0.117 |
| `Abs_r4` | 1024 | 8 | -0.097 ± 0.052 | -0.138 ± 0.115 | -0.060 ± 0.036 |
| `Abs_r4` | 1024 | 16 | -0.142 ± 0.120 | -0.173 ± 0.098 | -0.065 ± 0.033 |
| `Abs_r4` | 1024 | 32 | -0.211 ± 0.124 | -0.180 ± 0.104 | -0.179 ± 0.173 |

## Baselines

| arm | T=128 | T=1024 |
|---|---|---|
| `Signed_r4` | 1.000 ± 0.000 | 0.917 ± 0.020 |
| `Abs_r4` | 0.927 ± 0.089 | 0.558 ± 0.027 |

---

# Verdict against the pre-registration

## P1 — PASSES. The mechanism could operate here.

At the training length a mean of **8.2 of 64** channels (`Signed_r4`) and **17.0**
(`Abs_r4`) never complete a full cycle of their own period, so they see only a slice
of their phase range. At `T=1024` that falls to `0.8`–`4.3`: those channels are
exercised at phases they never saw in training. The precondition holds, so P2 is
live rather than void.

## P2 — REFUTED, and the sign is backwards. Consistently, on all four arms.

The prediction was that ablating LOW-frequency (under-trained) channels costs *less*
at `T=1024` than at `T=128`, because they are already being read out of
distribution. Accuracy change from ablating the `k=32` lowest-frequency channels:

| arm | T=128 | T=1024 | predicted |
|---|---|---|---|
| `Signed_r4` | $-0.001$ | $\mathbf{-0.139}$ | less negative at 1024 |
| `Vanilla_r4` | $-0.001$ | $\mathbf{-0.170}$ | less negative at 1024 |
| `Vanilla` (r=2) | $-0.030$ | $\mathbf{-0.254}$ | less negative at 1024 |
| `Abs_r4` | $-0.280$ | $-0.211$ | (floor artifact, see below) |

**The low-frequency channels are more load-bearing at OOD length, not less** — by
two orders of magnitude on the two clean arms, where ablating them is free at
training length and costs 0.14–0.17 at `T=1024`. The one arm that moves in the
predicted direction is `Abs_r4`, whose `T=1024` baseline is 0.558 against a measured
blank floor of 0.506; there is 0.05 of range to destroy, so that cell carries no
information.

Damage *is* localised in frequency — ablating high-frequency channels costs
0.42–0.57 at `k=32` against 0.14–0.25 for low — but in the opposite direction from
the account. The competing reading is simple: the low-frequency channels are the
ones with period at map scale, and coarse position matters more, not less, when the
walk covers more of the torus. That outweighs their being under-trained.

**The imported critical-dimension explanation does not survive.** It should be
removed from both documents as an explanation of the OOD-length signature.

## P3 — CONFIRMED, and it supplies the mechanism the refuted account was reaching for

`range(S)` at the two lengths, and the implied growth exponent `range ~ T^alpha`:

| arm | `T=128` | `T=1024` | ratio | **alpha** | opposition |
|---|---|---|---|---|---|
| `Signed_r4` | 91.84 | 269.65 | 2.94x | **0.518** | 0.125 |
| `Vanilla_r4` | 94.07 | 279.73 | 2.97x | **0.524** | 0.092 |
| `Vanilla` (r=2) | 88.47 | 320.54 | 3.62x | **0.619** | 0.495 |
| `Abs_r4` | 91.47 | 649.77 | 7.10x | **0.943** | 1.849 |

All four start from the same accumulator range at training length (88–94) and
separate entirely by how fast it grows. **A signed, well-conditioned increment gives
`alpha = 0.52` — a diffusive random walk. A monotone one gives `alpha = 0.94` —
ballistic drift.** `r=2` sits between at `0.62`.

The last column is the independently-measured opposition score
`||D(+x)+D(-x)|| / mean||D||`, from `SIGN_PROBE.md` and `ACTION_GEOMETRY.md`.
Across four arms from **two independent batches**, `r(opposition, alpha) = +0.9995`.

**One mechanism explains both the sign result and the rank result.** How well
opposite actions cancel sets whether the accumulator diffuses or drifts; that sets
how fast it leaves the range it was trained on; and that sets the rate of OOD
degradation. `r=2`'s skewed basis (opposition 0.495) and a monotone increment
(1.849) are the same failure at different severities.

Caveats: four points, correlational across arms, and `alpha ~ 0.5` for a signed
random walk is expected rather than surprising in itself — what carries the result
is the spread from 0.52 to 0.94 and its agreement with an opposition score measured
in a different batch for a different purpose. A direct test would bound the
accumulator and check that OOD degradation falls with it; the InEKF's wrap does
exactly that and is the natural arm.
