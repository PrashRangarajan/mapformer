# Loop x hierarchy: is the pairing free?

Parity trained at L=512, where the compute saving is real (-22.8% time, -19.9% memory at L=2048; see LOOP_HIER_COMPUTE.md).
A NULL ON ACCURACY IS THE SUCCESS CASE: the claim is cheaper at equal
accuracy, not better.

Parity is a TREE REDUCTION: parity(a,b,c,d) = parity(parity(a,b),
parity(c,d)). The partial parity of a pooled pair is EXACTLY a sufficient
statistic, so this is the case this project's standing principle --
hierarchy helps only when a summary is sufficient -- says should WIN, on a
task where hierarchy usually loses (exact recall).

Exact parameter parity within each row. Hourglass variants ignore
--n-layers and are always the 3-block scaffold, so the flat controls are
3-layer models, not 1-layer ones.

| arm | params | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|
| unshared, FLAT | 596,034 | 0.936 | 0.889 | 0.806 |
| unshared, HIER | 596,034 | 0.995 | 0.948 | 0.832 |
| SHARED, FLAT | 199,490 | 0.891 | 0.813 | 0.699 |
| SHARED, HIER | 199,490 | 0.962 | 0.897 | 0.800 |

## Hierarchy minus flat, per row

| row | L=512 | L=1024 | L=2048 |
|---|---|---|---|
| unshared | +0.060 (sd 0.093, MDE 0.075, 12/12) | +0.059 (sd 0.186, MDE 0.150, 8/12) | +0.026 (sd 0.215, MDE 0.174, 7/12) |
| shared | +0.071 (sd 0.100, MDE 0.081, 11/12) | +0.085 (sd 0.218, MDE 0.176, 9/12) | +0.101 (sd 0.285, MDE 0.230, 9/12) |

## Length decay -- the sharper half of the prediction

Accuracy above chance at L, as a fraction of that at the training length. A tree
reduction should decay FLATTER, not merely start higher.

| arm | L=512 | L=1024 | L=2048 |
|---|---|---|---|
| unshared, FLAT | 1.00 | 0.89 | 0.70 |
| unshared, HIER | 1.00 | 0.90 | 0.67 |
| SHARED, FLAT | 1.00 | 0.80 | 0.51 |
| SHARED, HIER | 1.00 | 0.86 | 0.65 |

## Verdict

**1. THE PAIRING IS FREE.** LoopedHourglass at 199,490 parameters matches the
unshared flat scaffold at 596,034 on accuracy -- +0.026 / +0.009 / -0.006 at
L=512/1024/2048, none of them detectable by either test -- while using **66.5%
fewer parameters, 22.8% less time and 19.9% less memory** (measured, not inferred:
LOOP_HIER_COMPUTE.md). Both savings, no accuracy cost. This is an equivalence read
off a null, so the MDE of 0.180 at L=2048 is the claim's real precision: the cost
is smaller than that, not zero.

**2. HIERARCHY DOES HELP HERE -- and the t-rule alone would have missed it.**
At the training length, hierarchy minus flat:

| row | effect | t | seeds | sign-test p |
|---|---|---|---|---|
| unshared, L=512 | +0.060 | +2.23 | **12/12** | **0.0005** |
| shared, L=512 | +0.071 | +2.44 | 11/12 | **0.006** |

Both fall under this project's t > 2.8 bar and would have been filed as
"unmeasured". Twelve out of twelve seeds in the same direction is p = 0.0005 by a
sign test. The MDE rule is conservative for a small consistent paired effect;
report both, and never treat t < 2.8 as evidence of absence when the sign count is
lopsided. At L=1024 and L=2048 the effect washes out by both tests.

**3. THIS CORRECTS MY OWN CONCLUSION FROM `HIER_PARITY.md`.** That run trained at
L=16, found +0.001 with an MDE of 0.006, and I wrote that the sufficient-statistic
principle "is not predictive" and should be retired as a predictor. **That test was
designed at a length where the mechanism cannot operate**: pooling k=2 over 16
tokens leaves 8 coarse tokens, so there is nothing to compress and no depth to save.
At L=512 -- 256 coarse tokens -- hierarchy helps in both rows on 23 of 24 seeds.

The tight null at L=16 was an artifact of my test design, not a refutation of the
principle. The principle survives, with a condition it did not previously carry:
**the summary must be sufficient AND the sequence long enough for compression to
mean anything.** The navigation losses it was built to explain are still exact-recall
failures; this adds that a short sequence is a second way for hierarchy to buy
nothing.

## Scope

One task, one pooling factor (k=2), one train length, n=16. Copy is not included: it has no dynamic range at any length here, so it cannot serve as a control (see ALGORITHMIC_RESULTS.md). Hierarchy is NOT combined with the loop in this run -- no looped-hierarchical variant exists, and building one is the natural follow-up if this is positive.
