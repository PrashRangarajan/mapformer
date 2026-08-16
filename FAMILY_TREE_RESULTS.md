# Family-tree (non-commutative relational structure) — results

The task MapFormer's appendix B.2.2 motivates and never runs: mother and
father do not commute. 8 relational actions, scored at revisited nodes,
n=3 seeds. Trained at 64 steps, also evaluated at 128 (OOD length).

**chance 0.125; effective floor is the HUB baseline 0.163** — shallow nodes
are revisited more, so answering with the most-visited node's observation
already scores 0.163. Read every number against 0.163.

| variant | n_steps=64 (train) | n_steps=128 (OOD) |
|---|---|---|
| MapEM-NC-L (non-commutative, linear) | 0.720 ± 0.011 | 0.671 ± 0.006 |
| MapEM-NC-NL (non-commutative, MLP) | 0.729 ± 0.010 | 0.672 ± 0.012 |
| MapEM single-p0 (COMMUTATIVE control) | 0.715 ± 0.008 | 0.659 ± 0.014 |
| Plain-Flat (index position, no PI) | 0.600 ± 0.011 | 0.550 ± 0.031 |

## Per seed (n_steps=64)

| variant | s0 | s1 | s2 |
|---|---|---|---|
| MapEM-NC-L (non-commutative, linear) | 0.713 | 0.733 | 0.714 |
| MapEM-NC-NL (non-commutative, MLP) | 0.720 | 0.740 | 0.726 |
| MapEM single-p0 (COMMUTATIVE control) | 0.712 | 0.725 | 0.709 |
| Plain-Flat (index position, no PI) | 0.589 | 0.610 | 0.603 |

## Paired analysis and verdict

| comparison | per-seed deltas | mean | wins |
|---|---|---|---|
| NC-L − commutative | +0.001 / +0.008 / +0.005 | **+0.005** | 3/3 |
| NC-NL − commutative | +0.008 / +0.016 / +0.016 | **+0.014** | 3/3 |
| commutative − index | +0.123 / +0.115 / +0.106 | **+0.115** | 3/3 |

**Non-commutativity buys almost nothing.** On a structure whose measured
non-commutativity is exactly 1.000 -- mother-then-father and father-then-mother
land on different people for EVERY node -- the commutative SO(2) model scores
0.715 against the non-commutative variants' 0.720 and 0.729. NC-NL is ahead on
3/3 seeds but by +0.014, about one standard deviation. NC-L by +0.005.

The paper's argument (appendix B.2.2) is that W_mother W_father != W_father
W_mother, so a commutative group CANNOT represent a family tree. That is correct
GROUP THEORY and it does not translate into task performance here.

**Path integration is the axis that matters: +0.115 over index position**, 23x
the non-commutativity effect. This corroborates Match-Query's finding on a
completely unrelated structure.

### The recurring error, stated plainly

This is the third measurement today where a mathematically real architectural
property failed to predict performance:

1. `A_P` kernel geometry — separate q0/k0 gives a provably non-peaked kernel, yet
   the configuration with the WORST kernel (100% negative at zero displacement)
   is the one where EM beats WM on 3/3 seeds. (`AP_KERNEL_DIAGNOSTIC.md`)
2. theta drift — a model WITH a working cognitive map scores worse on the
   path-integration "faithfulness" metric than one without. (`LAP_TRANSFER_NOREWARD.md`)
3. non-commutativity — exact group representation is provably required, and a
   model that cannot do it matches one that can. (this file)

The error each time was treating REPRESENTATIONAL NECESSITY as PERFORMANCE
NECESSITY. Models have other routes: content attention, approximate codes, and
outright memorisation.

### Caveats

63 nodes, sequences of 64/128, n=3. A 63-node tree may be largely memorisable by
content attention in a 128-dim model, which would let any position code coast.
**The test that would give NC a fair chance is a deeper tree** where node count
exceeds what attention can memorise. Not run.
