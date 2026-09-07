# The sign ablation: may the phase increment be negative?

Axis A5 of the relational map -- the one axis no paper varies deliberately.
Every content-dependent-phase mechanism outside MapFormer is non-negative,
and in every case that is a side-effect of a squashing function: CARoPE's
`1/(softplus+1)` lands in the OPEN interval (0,1), GRAPE-AP requires
`omega = g(x) >= 0`, CoPE's gate is a sigmoid. MapFormer's `Delta` is signed
only because nothing squashes it.

**Mechanism under test.** Non-negativity makes the per-channel angle
monotone in t. Monotone is all a language clock needs; a cognitive map needs
east and west to cancel. A monotone code can encode `(n_E, n_W)` but not
`n_E - n_W`, and revisit retrieval needs the latter.

Clean torus paper task, held-out map (env-seed 10000), one batch, 300 epochs
warmup+cosine at lr 1e-3. Parameter count identical across all five
MapFormer arms; the constraint adds nothing and does not change the rank.

## Accuracy (mean +/- sd)

| arm | Delta | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| `Signed_r4` | `W_out W_in x` (baseline) | 1.000 +/- 0.000 | 0.978 +/- 0.014 | 0.922 +/- 0.027 |
| `Vanilla_r4` | `W_out W_in x` (RNG control) | 1.000 +/- 0.000 | 0.984 +/- 0.009 | 0.927 +/- 0.015 |
| `Abs_r4` | `\|W_out W_in x\|` **primary** | 0.946 +/- 0.070 | 0.675 +/- 0.040 | 0.558 +/- 0.027 |
| `Pos_r4` | `softplus(.)` GRAPE-AP | 0.977 +/- 0.035 | 0.798 +/- 0.080 | 0.584 +/- 0.080 |
| `CARoPE_r4` | `1/(softplus(.)+1)` CARoPE | 0.900 +/- 0.138 | 0.809 +/- 0.133 | 0.645 +/- 0.090 |
| `RoPE` | index (floor) | 0.799 +/- 0.018 | 0.449 +/- 0.083 | 0.345 +/- 0.118 |

## Convergence (rule 9)

| arm | mean final loss | per-seed range |
|---|---|---|
| `Signed_r4` | 0.0002 | 0.0000 – 0.0004 |
| `Vanilla_r4` | 0.0002 | 0.0000 – 0.0003 |
| `Abs_r4` | 0.1708 | 0.0048 – 0.6112 |
| `Pos_r4` | 0.0676 | 0.0007 – 0.2949 |
| `CARoPE_r4` | 0.2929 | 0.0082 – 1.1226 |
| `RoPE` | 0.7844 | 0.6762 – 0.9140 |

- r(final loss, accuracy) at T=128: **-0.978** over 72 runs
- r(final loss, accuracy) at T=512: **-0.856** over 72 runs
- r(final loss, accuracy) at T=1024: **-0.729** over 72 runs

## Contrasts

Negative = first arm WORSE. DETECTABLE means |mean| > MDE = 2.8*sd/sqrt(n).

### T=128

| contrast | raw | loss-matched | verdict |
|---|---|---|---|
| `Abs_r4` - `Signed_r4` <br><sub>PRIMARY -- sign removed, nothing else</sub> | -0.054 (sd 0.070, t -2.67, MDE 0.057, 10/12 neg) | -0.006 (sd 0.022, t -0.95, MDE 0.018, 5/12 neg) | UNMEASURED |
| `Pos_r4` - `Signed_r4` <br><sub>GRAPE-AP style (init confound)</sub> | -0.023 (sd 0.035, t -2.32, MDE 0.028, 5/12 neg) | -0.004 (sd 0.012, t -1.22, MDE 0.010, 4/12 neg) | UNMEASURED |
| `CARoPE_r4` - `Signed_r4` <br><sub>CARoPE verbatim (init confound)</sub> | -0.100 (sd 0.138, t -2.50, MDE 0.112, 11/12 neg) | -0.017 (sd 0.028, t -2.05, MDE 0.023, 5/12 neg) | UNMEASURED |
| `Vanilla_r4` - `Signed_r4` <br><sub>RNG/construction control -- expect ~0</sub> | +0.000 (sd 0.000, t +1.00, MDE 0.000, 0/12 neg) | +0.000 (sd 0.000, t +0.69, MDE 0.000, 4/12 neg) | UNMEASURED |
| `Abs_r4` - `RoPE` <br><sub>does the monotone clock beat an index code?</sub> | +0.146 (sd 0.068, t +7.45, MDE 0.055, 1/12 neg) | -0.028 (sd 0.032, t -3.03, MDE 0.025, 10/12 neg) | **DETECTABLE NEGATIVE** |
| `Signed_r4` - `RoPE` <br><sub>the position effect, for scale</sub> | +0.201 (sd 0.018, t +39.06, MDE 0.014, 0/12 neg) | -0.021 (sd 0.016, t -4.68, MDE 0.013, 11/12 neg) | **DETECTABLE NEGATIVE** |

### T=512

| contrast | raw | loss-matched | verdict |
|---|---|---|---|
| `Abs_r4` - `Signed_r4` <br><sub>PRIMARY -- sign removed, nothing else</sub> | -0.303 (sd 0.044, t -23.83, MDE 0.036, 12/12 neg) | -0.215 (sd 0.068, t -11.01, MDE 0.055, 12/12 neg) | **DETECTABLE NEGATIVE** |
| `Pos_r4` - `Signed_r4` <br><sub>GRAPE-AP style (init confound)</sub> | -0.180 (sd 0.086, t -7.23, MDE 0.070, 12/12 neg) | -0.145 (sd 0.052, t -9.64, MDE 0.042, 12/12 neg) | **DETECTABLE NEGATIVE** |
| `CARoPE_r4` - `Signed_r4` <br><sub>CARoPE verbatim (init confound)</sub> | -0.169 (sd 0.132, t -4.43, MDE 0.107, 12/12 neg) | -0.018 (sd 0.088, t -0.69, MDE 0.071, 9/12 neg) | UNMEASURED |
| `Vanilla_r4` - `Signed_r4` <br><sub>RNG/construction control -- expect ~0</sub> | +0.006 (sd 0.020, t +0.97, MDE 0.016, 5/12 neg) | +0.006 (sd 0.020, t +0.97, MDE 0.016, 5/12 neg) | UNMEASURED |
| `Abs_r4` - `RoPE` <br><sub>does the monotone clock beat an index code?</sub> | +0.226 (sd 0.073, t +10.66, MDE 0.059, 0/12 neg) | -0.092 (sd 0.128, t -2.49, MDE 0.103, 9/12 neg) | UNMEASURED |
| `Signed_r4` - `RoPE` <br><sub>the position effect, for scale</sub> | +0.529 (sd 0.090, t +20.31, MDE 0.073, 0/12 neg) | +0.123 (sd 0.093, t +4.58, MDE 0.075, 0/12 neg) | DETECTABLE POSITIVE |

### T=1024

| contrast | raw | loss-matched | verdict |
|---|---|---|---|
| `Abs_r4` - `Signed_r4` <br><sub>PRIMARY -- sign removed, nothing else</sub> | -0.363 (sd 0.030, t -41.45, MDE 0.025, 12/12 neg) | -0.280 (sd 0.076, t -12.78, MDE 0.061, 12/12 neg) | **DETECTABLE NEGATIVE** |
| `Pos_r4` - `Signed_r4` <br><sub>GRAPE-AP style (init confound)</sub> | -0.338 (sd 0.087, t -13.51, MDE 0.070, 12/12 neg) | -0.305 (sd 0.070, t -15.02, MDE 0.057, 12/12 neg) | **DETECTABLE NEGATIVE** |
| `CARoPE_r4` - `Signed_r4` <br><sub>CARoPE verbatim (init confound)</sub> | -0.276 (sd 0.096, t -9.93, MDE 0.078, 12/12 neg) | -0.134 (sd 0.146, t -3.18, MDE 0.118, 10/12 neg) | **DETECTABLE NEGATIVE** |
| `Vanilla_r4` - `Signed_r4` <br><sub>RNG/construction control -- expect ~0</sub> | +0.006 (sd 0.031, t +0.65, MDE 0.025, 7/12 neg) | +0.006 (sd 0.031, t +0.65, MDE 0.025, 7/12 neg) | UNMEASURED |
| `Abs_r4` - `RoPE` <br><sub>does the monotone clock beat an index code?</sub> | +0.213 (sd 0.121, t +6.12, MDE 0.097, 0/12 neg) | -0.085 (sd 0.173, t -1.71, MDE 0.140, 9/12 neg) | UNMEASURED |
| `Signed_r4` - `RoPE` <br><sub>the position effect, for scale</sub> | +0.576 (sd 0.130, t +15.34, MDE 0.105, 0/12 neg) | +0.195 (sd 0.133, t +5.10, MDE 0.107, 0/12 neg) | DETECTABLE POSITIVE |

## Verdict against the pre-registration

**H1 fires at OOD LENGTH ONLY -- NOT the predicted result.** The pre-registration names this case explicitly: 'helps at OOD length' is this project's universal signature (rank, the InEKF, the forget gate and PoPE all show it and nothing explains it). An OOD-only sign effect is another instance of that unexplained axis, NOT evidence for the net-displacement mechanism.

**Control.** `Vanilla_r4 - Signed_r4` at T=512: +0.006 (sd 0.020, t +0.97, MDE 0.016, 5/12 neg) -> UNMEASURED. These two arms are mathematically identical and differ only in how many draws they take from the RNG, so anything other than UNMEASURED here means the batch is not readable.

## Degradation with length (H3, reported not adjudicated)

| arm | T=128 | T=1024 | drop |
|---|---|---|---|
| `Signed_r4` | 1.000 | 0.922 | -0.078 |
| `Vanilla_r4` | 1.000 | 0.927 | -0.073 |
| `Abs_r4` | 0.946 | 0.558 | -0.387 |
| `Pos_r4` | 0.977 | 0.584 | -0.393 |
| `CARoPE_r4` | 0.900 | 0.645 | -0.255 |
| `RoPE` | 0.799 | 0.345 | -0.454 |


---

# Reading it honestly

The auto-generated verdict above fired the "OOD LENGTH ONLY" branch, which the
pre-registration names as **not** the predicted result. That branch is mechanically
correct and its rule was written in advance, so it stands as written. But the rule
had a defect that only the data exposes, and two other pre-registered readings did
fire. All three go here.

## 1. The accuracy discriminator could not discriminate, and that is my design error

The pre-registered test was: a deficit at **training length** as well as OOD means
representational; OOD-only means this project's unexplained "helps at OOD length"
signature. At T=128 the baseline is **1.000 +/- 0.000**. There is a hard ceiling
with 0.054 of headroom against an MDE of 0.057. That cell cannot show a deficit of
any size, so requiring a signal there was requiring something the design cannot
deliver -- exactly the rule-11 ceiling trap this project already has written down,
built into my own verdict rule.

Do not read "not the predicted result" as "the mechanism is wrong". Read it as
**the accuracy discriminator was not measurable**, and go to the other two.

## 2. The training-length effect is real. It is in the LOSS, not the accuracy

| arm | mean final training loss | paired vs `Signed_r4` |
|---|---|---|
| `Signed_r4` | 0.0002 | — |
| `Vanilla_r4` | 0.0002 | -0.0000 (5/12) |
| `Abs_r4` | 0.1708 | **+0.1706 (12/12 worse)** |
| `Pos_r4` | 0.0676 | **+0.0674 (12/12 worse)** |
| `CARoPE_r4` | 0.2929 | **+0.2927 (12/12 worse)** |
| `RoPE` | 0.7844 | +0.7842 (12/12) |

Every constrained arm fits strictly worse than the signed baseline on **12 of 12
seeds**, at training length, at identical parameter count. The constraint bites
where the code is formed. Accuracy simply cannot report it at 1.000.

This also explains why the loss-matched contrast at T=128 is -0.006: `Abs` cannot
*reach* the baseline's loss (0.171 against 0.0002, ~1000x), so matching on loss
partials out the very quantity the constraint causes. At T=128, r(loss, acc) =
-0.978, so by rule 9 the loss is the only informative variable there anyway.

## 3. The probe fires exactly as pre-registered, and it is mechanism evidence

`SIGN_PROBE.md`, opposition score `||D(+x) + D(-x)|| / mean||D||` -- 0 means
opposite actions cancel exactly, 2 means they are identical:

| arm | oppose_x | oppose_y | \|cos(N,E)\| |
|---|---|---|---|
| `Signed_r4` | **0.125** | **0.106** | 0.218 |
| `Vanilla_r4` | **0.128** | **0.130** | 0.133 |
| `Abs_r4` | 1.849 | 1.855 | 0.587 |
| `Pos_r4` | 1.905 | 1.885 | 0.723 |
| `CARoPE_r4` | **1.981** | **1.977** | **0.930** |

The probe's pre-registered reading, verbatim: *"If the signed arm's opposition
scores are near 0 and the constrained arms' are near 2, the mechanism is confirmed
at the level of the learned code, independently of accuracy."* They are 0.11-0.13
and 1.85-1.98. **CARoPE's parameterisation reaches 1.98 of a possible 2.00** -- its
east and west are all but the same vector -- and its N/E axes collapse onto each
other (|cos| 0.93 against the signed arm's 0.13-0.22). The monotone models do not
merely perform worse; they demonstrably cannot represent a -1 action, and the
signed one demonstrably does.

## 4. The sharpest statement, which depends on none of the above

At **matched training loss**:

| contrast | T=128 | T=512 | T=1024 |
|---|---|---|---|
| `Signed_r4` - `RoPE` | -0.021 | **+0.123** | **+0.195** |
| `Abs_r4` - `RoPE` | **-0.028** | -0.092 (unmeas.) | -0.085 (unmeas.) |

Signed path integration beats an index code at OOD length, detectably, on 12/12
seeds. **Monotone path integration does not beat an index code anywhere** -- it is
detectably *worse* at T=128 and unmeasured elsewhere. So the entire measured value
of a content-dependent phase over a plain index clock, on this task, requires the
increment to be signed. Take the sign away and the mechanism is worth nothing over
RoPE.

## 5. Scope, per the amendment

This is a **replication in a new regime, not a discovery**, and the amendment
recorded that before the first checkpoint was read. Sarrof et al. observed the
non-negativity, Grazzi et al. proved positive-only eigenvalues cannot do parity and
fixed it, and Selective RoPE's 4.2 already demonstrated single-layer Transformer
parity from input-dependent rotations that "model flips". What is new here is only
the regime (navigation rather than parity and formal languages) and the isolation
(`|D|` against `D`, one operation, identical parameter count, 12 seeds, with an
RNG-path control that lands at +0.006).

The control behaved: `Vanilla_r4 - Signed_r4` is +0.000/+0.006/+0.006, UNMEASURED
at every length, so the double RNG draw in the constrained arms is not the story.
