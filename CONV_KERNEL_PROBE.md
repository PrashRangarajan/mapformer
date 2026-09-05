# What the causal conv learned

Selective RoPE's short-convolution is derived in its own Sec. 3 from the
shift `q_t - q_{t-1}` -- a **first difference**, kernel width 2. A first
difference inside a cumsum telescopes to `u_t - u_0`, so the exact
difference kernel removes the accumulation entirely. The K axis is
therefore not sharp-vs-blurred; it runs from **accumulation** (identity
kernel) to **no accumulation** (differencer).

Per-channel projections of the unit-normalised learned kernel.

| arm / task | \|·identity\| | \|·differencer\| | \|DC gain\| | n |
|---|---|---|---|---|
| ConvAngle, parity | 0.445 | 0.385 | 0.781 | 16 |
| SRoPEGen, parity | 0.423 | 0.418 | 0.777 | 16 |
| ConvAngle, torus | 0.470 | 0.345 | 1.063 | 8 |
| SRoPEGen, torus | 0.460 | 0.463 | 0.822 | 8 |

Reference points:

| kernel | \|·identity\| | \|·differencer\| | \|DC gain\| |
|---|---|---|---|
| random kernel (uniform on S^3) | 0.424 | 0.424 | 0.798 |
| identity  [0, 0, 0, 1] | 1.000 | 0.707 | 1.000 |
| differencer [0, 0, -.7, +.7] | 0.707 | 1.000 | 0.000 |

## Verdict

**The learned kernels sit at the random baseline on every metric.** For a
unit vector drawn uniformly on the 3-sphere the expected absolute
projection onto any fixed unit direction is 0.424; the measured values are
0.423--0.470 against identity and 0.345--0.463 against the differencer.

So the conv learns **neither** of the two structured kernels available to
it. It does not become an identity (which would make it harmless, leaving
the path integral intact) and it does not become the first difference its
own derivation motivates. It stays an essentially arbitrary smear of the
increment applied before accumulation, which is a sufficient account of why
it costs on both tasks (-0.020 parity, -0.064 torus) for 193 parameters --
the only knob in `SELECTIVE_ROPE.md` that is free and negative on both.

The one visible tendency is small and in the expected direction: the torus
`ConvAngle` kernels lean slightly toward identity (0.470 vs 0.424) and
carry more DC (1.063 vs 0.798) than the parity ones, i.e. they try harder
to get out of the way of the cumsum on the task where the cumsum is the
thing being computed. The margin is far too small to rest anything on.

## Scope

Our K=4 is an assumption from the Mamba/GLA convention (`d_conv=4`); the
paper states no width in its pseudocode, its two ablation tables, or its
hyperparameter appendices. **K=2 with a fixed difference kernel has not
been run**, and it is the setting their derivation actually specifies.
