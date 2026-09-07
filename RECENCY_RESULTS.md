# Recency (k-back): the clock/map crossover

6 arms x 8 seeds, ONE batch (rule 3). `k_max=64`, `p_filler=0.5`, `min_gap=64`,
train `T=1024`, 300 epochs cosine, lr 1e-3, 1 layer, d=128, fast-attn.
Chance **0.0625**; most-recent shortcut floor **0.0771**. Gates:
`RECENCY_GATES_K64.md` (all rows PASS at 800 episodes).
Hypotheses and the pre-launch amendment: `RECENCY_PREREG.md`.

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Signed_r4` (unconstrained) | 0.025 | **1.000 +/- 0.000** | 0.940 +/- 0.050 |
| `CARoPE_r4` (monotone) | 0.020 | **1.000 +/- 0.000** | **0.973 +/- 0.028** |
| `Pos_r4` (monotone) | 0.061 | 0.979 +/- 0.041 | 0.947 +/- 0.069 |
| `Abs_r4` (monotone) | 0.110 | 0.961 +/- 0.068 | 0.889 +/- 0.126 |
| `RoPE` (index) | 2.150 | **0.234 +/- 0.025** | 0.234 +/- 0.011 |
| `PlainFlat` (index) | 2.143 | **0.236 +/- 0.019** | 0.233 +/- 0.009 |

## 1. The headline: a fixed index code cannot do contextual counting

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| path-integrated - index, T=1024 | **+0.750** | 0.030 | 0.030 | 8/8 | **DETECTABLE** |
| path-integrated - index, T=2048 | **+0.704** | 0.057 | 0.056 | 8/8 | **DETECTABLE** |

**The largest effect measured anywhere in this project** -- the torus position
effect is +0.461. And the per-offset curve at T=1024 says exactly why:

| arm | k=1 | k=8 | k=16 | k=32 | k=48 | k=64 | slope/k |
|---|---|---|---|---|---|---|---|
| `Signed_r4` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00000 |
| `CARoPE_r4` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00000 |
| `Pos_r4` | 0.97 | 0.99 | 0.96 | 0.96 | 0.96 | 1.00 | +0.00012 |
| `Abs_r4` | 0.97 | 1.00 | 0.98 | 0.96 | 1.00 | 0.93 | -0.00009 |
| `RoPE` | **0.99** | 0.33 | 0.16 | 0.18 | 0.28 | **0.11** | -0.00374 |
| `PlainFlat` | **1.00** | 0.27 | 0.19 | 0.19 | 0.26 | **0.13** | -0.00414 |

The index arms solve `k=1` outright and collapse from `k=8` on. That is CoPE's
own account made quantitative -- arXiv:2405.11582, verified first-hand in
`papers/txt/cope.txt`: against contextual position, relative PE's "best it can do
is a decaying attention". Path-integrated arms are FLAT in k out to 64.

Note this contrast is confounded with training loss by construction (index 2.15
vs path 0.02-0.11) and loss-matching it is not meaningful: *the index arms cannot
fit the task* is the finding, not a nuisance.

## 2. H1 amended -- the cost of constraining to monotone is ~0 here

`T=1024` is at CEILING for two arms (1.000 +/- 0.000), so the primary contrast
moves to `T=2048` under the pre-registered ceiling clause.

| contrast (T=2048) | raw | MDE | loss-matched | MDE | verdict |
|---|---|---|---|---|---|
| `Abs_r4` - `Signed_r4` | -0.052 | 0.105 | +0.016 | 0.046 | unmeasured |
| `Pos_r4` - `Signed_r4` | +0.007 | 0.067 | +0.035 | 0.031 | detectable (1 of 3) |
| `CARoPE_r4` - `Signed_r4` | +0.032 | 0.060 | +0.028 | 0.047 | unmeasured |
| **mean(monotone) - `Signed_r4`** | **-0.004** | 0.055 | **+0.026** | 0.027 | **UNMEASURED** |

Against the torus, where the same contrast is **-0.215 / -0.280** at T=512/1024
on 12/12 seeds. **The interaction is ~ +0.28**: forcing a monotone increment
costs a fifth to a quarter of accuracy on a map task and nothing measurable on a
clock task. That is the pre-registered prediction, and it is the first
demonstration in this project that a mechanism's value is decided by the task
rather than being a property of the mechanism.

## 3. H2 (the primary claim) -- alpha is DIAGNOSTIC, not just descriptive

`Signed_r4` is unconstrained, so it may adopt either code. It adopts a different
one per task:

| arm | alpha (torus) | alpha (recency) | delta | se |
|---|---|---|---|---|
| **`Signed_r4`** (free) | **0.591 +/- 0.028** | **0.967 +/- 0.009** | **+0.376** | 0.009 |
| `Abs_r4` (constrained) | 1.010 | 0.976 | -0.033 | 0.007 |
| `Pos_r4` (constrained) | 1.005 | 1.005 | +0.000 | 0.002 |
| `CARoPE_r4` (constrained) | 1.003 | 0.992 | -0.011 | 0.001 |

n=12 torus, n=8 recency. **The same architecture learns a diffusive accumulator
(alpha 0.59, a MAP) on the torus and a ballistic one (alpha 0.97, a CLOCK) on
recency**, t ~ 42, while every arm that CANNOT choose sits at ~1.0 on both -- the
control that makes the shift attributable to the choice rather than to the task's
statistics.

This is what the clock/map dichotomy predicted and had never been tested on.

### How it does it -- ESTABLISHED causally (`RECENCY_GATE_ABLATION.md`)

The negative fraction of Delta barely moves (0.498 -> 0.478), so the arm does NOT
make every increment positive. It learns a **content gate**: large increments on
counted tokens, near-zero on filler. Per head that gate is present in **8/8**
seeds (an earlier mean-over-heads statistic said 6/8 and was the wrong statistic;
an earlier version still averaged SIGNED values across seeds, where theta's global
sign is arbitrary, and reported noise).

Correlation could not settle it -- gate strength does not predict the outcome,
r(log gate ratio, accuracy) = **-0.34**, across a range where every seed is near
ceiling and the correlation has no power either way. Settled by eval-only
intervention instead:

| condition | accuracy | delta |
|---|---|---|
| baseline | 1.000 | -- |
| `zero_filler` (filler increments removed) | **0.9997** | **-0.0003** |
| `zero_content` (counted increments removed) | 0.068 | -0.932 |
| `uniform_content` -- constant Delta on content, magnitude-matched | **0.783** | -0.217 |
| `uniform_all` -- constant Delta on every token, magnitude-matched | **0.189** | -0.811 |

1. **`zero_filler` is a no-op.** Filler increments carry nothing.
2. **`zero_content` destroys it.** The positive control bites, so the
   intervention reaches the pathway.
3. **`uniform_content` beats `uniform_all` by +0.594 at 8/8 seeds.** Both replace
   Delta with a constant and both make theta travel the same total distance; they
   differ only in WHICH tokens the constant lands on. That isolates the value of
   counting content rather than tokens.

A constant increment on content alone recovers 0.783 of 1.000, so everything the
model learned beyond "count the counted tokens" is worth ~0.22 -- **the gate is
most of the mechanism, not a part of it.** And `uniform_all` (0.189) lands BELOW
the index arms (0.234): a token clock inside this architecture is no better than
an index code, which is exactly what it is.

**A confound I nearly published.** My pre-stated criterion used an `equalize`
condition (filler increments set equal to content). It collapses to 0.086 --
apparently decisive. But `scale_match`, which changes ONLY theta's scale while
still counting content alone, collapses just as hard (0.110). This model is
acutely sensitive to theta's absolute magnitude, so `equalize` proves nothing and
is not counted. The magnitude-matched pair was added afterwards, as a control for
that confound rather than a second bite at the hypothesis, and was not
pre-registered.

## 4. Two of my four pre-registered hypotheses were REFUTED

- **H4 (index should do well on a clock task): refuted, sign inverted.** A
  CONTEXTUAL clock is precisely what a fixed index cannot encode. Recorded in the
  amendment before launch, after the pilots.
- **H3 (signed should decay in k, monotone flat): refuted.** BOTH are flat; the
  decay is in the index arms. The reason is H2 -- the signed arm learns a
  monotone code, so there is no signed-vs-monotone difference left to decay.

## Caveats, none of which are buried

- **Convergence (rule 10): CHECKED, and it holds.** The loss slope over the
  final 10% is -0.002 to -0.005/epoch and the flat fraction only 1/8 to 4/8 per
  arm, so the 300-epoch numbers were not licensed on their own. A 2x-budget rerun
  (600 epochs, 3 seeds) moves nothing: RoPE 0.251 -> 0.242, PlainFlat 0.247 ->
  0.248, `Signed_r4` 1.000 -> 1.000, index final loss 2.15 -> 2.12. **0.234 is a
  capability limit, not a budget limit**, so +0.750 is licensed rather than an
  upper bound.
- **r(final loss, accuracy) = -0.999** over all 48 runs and **-0.875** within the
  four path-integrated arms at T=2048 (rule 9). Section 2 is therefore reported
  loss-matched as well as raw; section 1 is not, for the reason given there.
- `T=1024` is at ceiling for two arms. The move to T=2048 was pre-registered, not
  chosen after seeing the table.
- One task, one `k_max`, one `p_filler`, n=8. The natural external-validity check
  is **Flip-Flop LM** (Liu et al. 2023), which is this task at k=1 and is a
  published dataset; it has not been run.
