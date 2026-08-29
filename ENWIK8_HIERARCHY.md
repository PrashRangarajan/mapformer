# Does the time-hierarchy help on TEXT? (enwik8, byte-level)

Generated from the run JSONs, not typed by hand -- an earlier enwik8 hierarchy
figure in CLAUDE.md ("2.00 vs 2.07, hierarchy better") was supported by no saved
data and had the SIGN WRONG.

**Setup.** 36k iters, seq 512, batch 16, lr 2e-4, dim 880, 9 layers, r=4, seed 42,
deterministic val (fixed generator -- before that fix, val noise of 0.02-0.07
REORDERED arms whose true spread was 0.011).

**The primary comparison is MapWM-Hier vs MapWM-FlatHG**: the same 3-block
Hourglass scaffold at identical parameter count, differing ONLY in whether the
middle block pools by k=2. Nothing else moves.

| model | params | final val bpc | best val bpc | wall | it/s |
|---|---|---|---|---|---|
| MapWM-Hier | 28,371,016 | 1.4537 | 1.4537 | 69.8 min | 8.6 |
| MapWM-FlatHG | 28,371,016 | 1.4506 | 1.4506 | 75.9 min | 7.9 |
| MapPoPE-Hier | 28,372,336 | 1.4591 | 1.4591 | 72.6 min | 8.3 |
| PoPE-Hier | 28,367,936 | 1.4553 | 1.4553 | 71.3 min | 8.4 |

## Verdict

Parameter parity is exact: 28,371,016 vs 28,371,016 (IDENTICAL).

**Pooling effect on bpc: +0.0032 final, +0.0032 best** (negative = hierarchy better).
Checkpoint-to-checkpoint sd after the determinism fix is 0.003-0.007, so an effect
of 0.0032 is INSIDE the noise. 
**Hierarchy is a null on next-byte prediction quality.**

**Pooling effect on cost: 1.23x throughput, -14.1% peak memory.** This is the only
measurable effect, and it is an EFFICIENCY effect.

| | MapWM-Hier | MapWM-FlatHG | effect |
|---|---|---|---|
| throughput, isolated on an idle GPU | 20.10 it/s | 16.36 it/s | **1.23x (-18.6% step time)** |
| peak memory | 2.62 GiB | 3.05 GiB | **-14.1%** |
| analytic block FLOPs | | | **-17.4%** |

**CORRECTION.** The first version of this file reported -8.6% wall time, taken from
the training run. That measurement was contaminated: both arms shared cuda:0 and
Hier finished 6 minutes earlier, leaving FlatHG a quieter GPU for its tail. Measured
alone on an idle GPU the saving is 1.23x, which matches the -17.4% analytic FLOP
count. Do not quote the 8.6%.

### Where the saving actually comes from -- NOT the quadratic term

The scaffold is 1 pre / 1 coarse / 1 post, so only ONE of three blocks pools. Per
block, the attention matmuls (2*L*d) sit against 12*d^2 of projections and FFN, and
at d=880, L=512 attention is only **8.8%** of a block. So the saving is overwhelmingly
"half as many tokens through one block's FFN" -- a LINEAR win -- not the quadratic
attention win the hourglass is usually sold on. It therefore barely scales:

| seq len | attention share of a block | hier/flat FLOPs |
|---|---|---|
| 512 | 8.8% | -17.4% |
| 2048 | 27.9% | -19.0% |
| 8192 | 60.8% | -21.7% |

The ceiling for this scaffold is **-25%**, reached only if attention dominated
entirely. Buying substantially more compute back needs a different scaffold -- more
or deeper coarse blocks, or a larger shorten factor -- not a longer sequence.

## This reproduces the earlier plain-family result, and fixes its sign in CLAUDE.md

| family | hier | flat | effect |
|---|---|---|---|
| plain (31,787,264 params, seq 2048) | 1.4844 | 1.4727 | +0.0117 hier worse |
| MapFormer (28,371,016 params, seq 512) | 1.4537 | 1.4506 | +0.0032 |

Same direction in both families -- hierarchy costs a little quality and saves
real compute. It is an efficiency property on text, NOT a quality win, and it
must not be listed among hierarchy's wins.

## The two PoPE arms are EXPLORATORY, not clean

They are reported for completeness only. `MapPoPE-Hier` silently trained at
**rank 2 while the MapWM arms trained at rank 4** -- `_widen_to_d` rebuilt
`action_to_lie` at the default rank and discarded `--bottleneck-r`. Found and
fixed mid-run (commit e8f8f50, KNOWN_BUGS.md); the fix does NOT apply
retroactively to these checkpoints, so any MapPoPE-vs-MapWM read here is
rank-confounded. `PoPE-Hier` is correctly rank-invariant (no action subspace).
**The primary pair is unaffected: both arms are MapWM, both r=4.**

There is no flat control for either PoPE arm, so this run says nothing about
whether hierarchy helps the PoPE family. It was never designed to.

## What this does and does not settle

Settles: on byte-level text at this scale, pooling buys throughput, not bpc,
in the MapFormer family as well as the plain one. n=1 seed per arm, so only an
effect much larger than 0.007 bpc would have been detectable -- consistent with
a null, not proof of one (standing rule 11).

Does not settle: whether hierarchy helps text tasks that reward COMPOSITIONAL
TRANSFER or LONG-HORIZON AGGREGATION, which is where its wins actually live
(compositional motif 0.415 vs flat 0.270; aggregate T=2048 0.537 vs 0.401).
Next-byte prediction is an exact-recall objective, the regime where a lossy
summary is NOT a sufficient statistic and hierarchy is expected to lose.
