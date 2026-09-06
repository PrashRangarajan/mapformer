# The "between" is r=4, and it costs 384 parameters

Selective RoPE's torus win came from its two ~8k-parameter knobs, which were
statistically indistinguishable from each other -- pointing at capacity or
conditioning rather than either mechanism. The gate-as-token-suppressor hypothesis
was tested directly and falsified (`GATE_PROBE.md`). This tests the remaining
candidate on the knob MapFormer already owns.

Torus, trained at T=128, 8 seeds, one batch, same recipe as the selective run so
the numbers are directly comparable.

| arm | params | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| Vanilla (r=2) | 204,373 | 0.993 ± 0.017 | 0.944 ± 0.028 | 0.834 ± 0.064 |
| **Vanilla_r4** | **204,757** | 1.000 ± 0.000 | 0.982 ± 0.005 | **0.919 ± 0.012** |
| Vanilla_r8 | 205,525 | 1.000 ± 0.000 | 0.981 ± 0.018 | 0.925 ± 0.033 |
| Vanilla_r16 | 207,061 | 1.000 ± 0.000 | 0.972 ± 0.015 | 0.913 ± 0.018 |
| Vanilla_r32 | 210,133 | 1.000 ± 0.000 | 0.982 ± 0.010 | 0.928 ± 0.021 |

Paired against r=2, T=1024: r4 **+0.085** (t 3.57, 8/8, sign p=0.008), r8 +0.091
(8/8), r16 +0.079 (8/8), r32 +0.095 (8/8). At T=512: +0.038 / +0.037 / +0.028 /
+0.038, r4 and r32 at 8/8.

## The answer to "something between MapFormer and Selective RoPE"

**There is nothing to build.** The cheapest fix is a rank the model already has:

| route to +0.086 at T=1024 | parameters |
|---|---|
| Selective RoPE's sigmoid gate | +8,193 |
| **r = 4** | **+384** |

**The same win for 21x fewer parameters, no new mechanism, and no conv to pay
for.** And the curve is a STEP, not a slope: r4 through r32 are flat within noise
(+0.085 to +0.095), so this is not "more capacity is better" -- it is "r=2
specifically is too tight".

## Why r=2 loses: the code it learns is skewed

`ACTION_GEOMETRY.md` reads the learned latents directly. Given four dimensions the
model puts its actions in a 2-plane anyway -- 100.0% of the spectral energy, and
still 99.96% at r=32 -- so the paper's dimensional claim is vindicated. What fails
is the code r=2 is forced into:

| arm | opposition (N+S~=0) | \|cos(N,E)\| | obs / action norm |
|---|---|---|---|
| r=2 | **0.495** | **0.783** | 0.139 |
| r=4 | **0.092** | **0.175** | 0.042 |

At r=2 opposite actions fail to cancel by half the action scale, and north and east
are nearly PARALLEL. At r=4 both are 4-5x cleaner and observations sit closer to
the origin. Squeezing the latent into exactly two dimensions does not produce a
clean displacement vector -- it produces a badly conditioned one, because the
optimiser cannot separate the axes inside so tight a space.

**That is the mechanism for the +0.085**: not missing capacity, a skewed basis.

**Interpretability is preserved, with one caveat.** The displacement reading the
paper relies on is recovered exactly by projecting the r=4 latents onto their top
two singular directions -- and from a cleaner model. What is lost is that the
projection becomes a step you perform rather than a property you get by
construction. If the latent must BE the displacement with no post-processing, r=2
remains the only option, and it costs 0.085 and a non-orthogonal basis.

## What this says about the paper's justification

App. A.7 justifies r=2 structurally: r is the dimensionality of the action space,
so on a 2D grid `Delta_in` is the displacement vector itself. That argument is
right about what is EXPRESSIBLE and wrong about what is TRAINABLE.

Both halves are demonstrated here. A rank-2 solution exists and is found routinely
-- `Vanilla` reaches 0.944 at T=512 -- so two dimensions are representationally
sufficient, exactly as the paper argues. But r=2 still costs **0.085 at T=1024**
against r=4, on 8 of 8 seeds. The bottleneck is not too small to express the
displacement; it is too tight to optimise through.

That also resolves the tension with `RANK_TRUNCATION.md`, where an unconstrained
map could not be projected below rank ~16 without collapsing. Trained-at-rank-r and
truncated-to-rank-r are different objects: r=4 trains fine, while a full-rank map
truncated to 4 scores 0.594. Post-hoc rank is not training-time rank.

**Recommended default: r=4.** It costs 0.19% more parameters and buys +0.085 at
8x the training length (trained at T=128, measured at T=1024; at 4x it is +0.038), with lower seed variance at every length (±0.012 against
±0.064 at T=1024).

## What is not tested

**r=3.** The step lands between 2 and 4, and r=3 would distinguish two accounts:
"2 displacement dimensions plus one action/observation axis" predicts r=3
suffices; "the optimiser just needs slack" predicts a smooth recovery. One arm,
eight runs.

Also untested: **r=1**, the only genuine test of the paper's dimensional argument,
since one dimension cannot span two independent axes. The claim that r=1 collapses
to 0.66 has no experiment behind it anywhere in this repository.

Torus only. Parity is a 1-dimensional task, where the argument predicts even less
rank sensitivity, and it was not run here.
