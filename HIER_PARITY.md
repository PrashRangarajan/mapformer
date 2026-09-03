# Does hierarchy help parity?

Parity is a TREE REDUCTION: parity(a,b,c,d) = parity(parity(a,b),
parity(c,d)). The partial parity of a pooled pair is EXACTLY a sufficient
statistic, so this is the case this project's standing principle --
hierarchy helps only when a summary is sufficient -- says should WIN, on a
task where hierarchy usually loses (exact recall).

Exact parameter parity within each row. Hourglass variants ignore
--n-layers and are always the 3-block scaffold, so the flat controls are
3-layer models, not 1-layer ones.

| arm | params | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|---|
| index, FLAT (3 layers) | 595,586 | 0.966 | 0.767 | 0.631 | 0.565 | 0.532 |
| index, HIERARCHICAL | 595,586 | 0.986 | 0.769 | 0.632 | 0.566 | 0.532 |
| path-int, FLAT (3 blocks) | 596,034 | 0.988 | 0.990 | 0.854 | 0.700 | 0.593 |
| path-int, HIERARCHICAL | 596,034 | 1.000 | 0.983 | 0.844 | 0.687 | 0.582 |

## Hierarchy minus flat, per row

| row | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|
| index | +0.020 (sd 0.070, MDE 0.049, 15/16) | +0.002 (sd 0.041, MDE 0.029, 5/16) | +0.001 (sd 0.021, MDE 0.014, 9/16) | +0.001 (sd 0.009, MDE 0.006, 9/16) | +0.000 (sd 0.005, MDE 0.004, 8/16) |
| path-int | +0.012 (sd 0.002, MDE 0.002, 16/16) | -0.007 (sd 0.049, MDE 0.034, 13/16) | -0.009 (sd 0.191, MDE 0.134, 10/16) | -0.014 (sd 0.156, MDE 0.109, 8/16) | -0.010 (sd 0.134, MDE 0.094, 8/16) |

## Length decay -- the sharper half of the prediction

Accuracy above chance at L, as a fraction of that at L=16. A tree
reduction should decay FLATTER, not merely start higher.

| arm | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|
| index, FLAT (3 layers) | 1.00 | 0.57 | 0.28 | 0.14 | 0.07 |
| index, HIERARCHICAL | 1.00 | 0.55 | 0.27 | 0.14 | 0.07 |
| path-int, FLAT (3 blocks) | 1.00 | 1.00 | 0.73 | 0.41 | 0.19 |
| path-int, HIERARCHICAL | 1.00 | 0.97 | 0.69 | 0.37 | 0.16 |

## Verdict

- **index** at L=128: +0.001, MDE **0.006** -> a TIGHT NULL, not an unmeasured one
- **path-int** at L=128: -0.014, MDE **0.109** -> genuinely underpowered

Those are different verdicts and must not be reported as one. In the index row the
effect is bounded at +/-0.006, far below anything worth caring about, so this is a
real null: hierarchy does not help. In the path-integrated row the MDE is 0.109 and
nothing smaller than that could have been seen, so it says nothing either way.

**The tree-reduction prediction FAILED where it was sharpest.** Parity is literally
`parity(a,b,c,d) = parity(parity(a,b), parity(c,d))`; the pooled summary is exactly
sufficient with zero loss; and hierarchy still buys nothing at any extrapolation
length. The length-decay curves are indistinguishable -- index 0.57/0.28/0.14/0.07
flat against 0.55/0.27/0.14/0.07 hierarchical -- when the whole argument for
hierarchy here was that a log-depth tree should decay FLATTER than a linear scan.

**So this project's standing principle is not predictive.** "Hierarchy helps when a
lossy summary is a sufficient statistic" was built to explain the navigation
losses, and it has now been tested at its single most favourable case -- lossless
summary, tree-structured algorithm, length extrapolation -- and produced a tight
null. Sufficiency of the summary is evidently NECESSARY but not SUFFICIENT for
hierarchy to pay. Do not cite the principle as a predictor; cite it, if at all, as
a description of where hierarchy has been observed to fail.

**The one thing hierarchy does buy is at TRAINING LENGTH, and it is small:** +0.020
index (15/16) and +0.012 path-int (16/16, MDE 0.002 -- very tight). Consistent,
detectable, and gone by L=32.

**Where that leaves hierarchy relative to the other two mechanisms.** Path
integration buys +0.078 for 448 parameters and the loop buys +0.056 for none, both
on this same task. Hierarchy buys nothing here at 3x the parameters. Its record
elsewhere -- compositional transfer, and compute at equal quality on text -- is on a
different axis from either, and nothing in this run suggests it joins their stack.

## Scope

One task, one pooling factor (k=2), one train length, n=16. Copy is not included: it has no dynamic range at any length here, so it cannot serve as a control (see ALGORITHMIC_RESULTS.md). Hierarchy is NOT combined with the loop in this run -- no looped-hierarchical variant exists, and building one is the natural follow-up if this is positive.
