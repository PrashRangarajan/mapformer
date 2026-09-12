# PAIRCONST -- results (pre-registration `PAIRCONST_PREREG.md`, commit 6d3d99a)

The capacity control for PAIRORIGIN. `EMPairConst_r4` is EMPair's pathway reading a LEARNED
CONSTANT, so origins are identical for every token and the kernel stays shared -- with 224,537
parameters against EMPair's 224,409, i.e. **128 MORE**.

Manipulation checks, both PASS: the determinism re-check is bitwise identical (29/29 tensors,
loss curves equal), and the origin spread across tokens is exactly **0.000** for EMPairConst
against **0.117** for EMPair.

## Accuracy: BOTH registered verdicts fail

| arm | acc T=1024 | acc T=2048 | final loss |
|---|---|---|---|
| `EMPair_r4` | 0.880 +/- 0.136 | 0.784 | 0.422 |
| **`EMPairConst_r4`** | **0.782 +/- 0.159** | 0.659 | 0.755 |
| `VanillaEM_P0_r4` | 0.600 +/- 0.126 | 0.510 | 1.340 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EMPair - EMPairConst (C1: freedom) | +0.098 | 0.211 | 0.209 | 7/8 | unmeasured |
| EMPairConst - P0 (C2/C3: parameters) | +0.182 | 0.190 | 0.188 | 6/8 | unmeasured |
| EMPair - P0 (the effect under test) | +0.280 | 0.218 | 0.216 | 7/8 | DETECTABLE |

- **C1 NOT CONFIRMED** (+0.098 needs >= +0.20 and detectable).
- **C2 "MET" by the letter and NOT by the spirit.** +0.182 is below its MDE of 0.188 by 0.006.
  Reporting "the parameters buy nothing" would be false precision: this is **unmeasured**, with
  a point estimate two thirds the size of the whole effect.
- **C3 not confirmed.** As the pre-registration required when both fail: **unresolved.**

**So PAIRORIGIN's +0.280 splits roughly in half -- ~+0.18 from the extra pathway and ~+0.10 from
content-dependence -- and at n=8 neither half is individually detectable.**

## Mechanism: the attribution IS clean, and it goes to freedom

| arm | phase spread across pairs | solved cells via a per-token rewind | solved cells (k>=8) |
|---|---|---|---|
| `VanillaEM_P0_r4` | 0.000 | 0.964 | 23.5 |
| **`EMPairConst_r4`** | **0.000** | **0.948** | 36.8 |
| `EMPair_r4` | 1.448 | **0.189** | 30.8 |

The control has MORE parameters than EMPair, trains the same pathway, and **keeps the per-token
rewind route** (0.948, indistinguishable from single-`p0`'s 0.964). Only the arm with
content-dependent origins abandons it (0.189).

**Therefore: what abandons the per-token rewind is per-pair freedom, not capacity.** That part of
`PAIRORIGIN_RESULTS.md` stands. What does NOT stand is the clean attribution of the ACCURACY
gain: extra capacity in the position pathway improves accuracy by ~+0.18 while leaving the
mechanism untouched, which is a real and separate effect this line had not measured.

## What would resolve the accuracy split

At n=8 the MDE is 0.209 against a +0.098 effect. Detecting it needs roughly **n = 36** per arm
(MDE scales as 1/sqrt(n)), i.e. ~72 runs, about 3 GPU-hours. Worth doing only if the split
matters for a claim; the mechanism question it was built to answer is already settled by the
route readout, which does not depend on the accuracy contrast at all.

## Correction this forces upstream

`PAIRORIGIN_RESULTS.md` and the summary blocks recorded "the decisive test of the kernel-sharing
claim FIRES". That is now too strong on one of its two legs and is corrected in place:

- **Mechanism (stands):** per-pair freedom, not capacity, is what stops the model rewinding
  query tokens -- established by a control with more parameters that does not stop.
- **Accuracy (does not stand as attributed):** the +0.280 is real and detectable, but how much
  of it is freedom versus pathway capacity is UNRESOLVED at n=8.
