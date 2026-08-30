# enwik8 — what does HIERARCHY buy on top of path integration / PoPE?

36k iters, deterministic val, seq 512, dim 880 for all hierarchical arms. Mean of last 5 checkpoints. **Lower is better.** n=1 (shape first).

## Primary: pooling isolated (parameter-IDENTICAL pair)

| model | pooling | params | val bpc |
|---|---|---|---|
| MapWM-Hier | ON (k=2) | 28,371,016 | 1.4609 |
| MapWM-FlatHG | OFF | 28,371,016 | 1.4598 |

**Effect of pooling: +0.0010** (negative = hierarchy helps).
Same isolation that gave +0.130 on the compositional task (n=8).

## Exploratory: the other hierarchical arms

| model | params | val bpc |
|---|---|---|
| MapPoPE-Hier | 28,372,336 | 1.4657 |
| PoPE-Hier | 28,367,936 | 1.4610 |

### Flat arms at 36k for reference (dim 512, 28.6M, NOT param-matched to the
### hier arms -- they are 0.96% larger, and the PoPE hier arms also differ in rank)

| model | val bpc |
|---|---|
| RoPE | 1.3864 |
| PoPE-Flat | 1.3806 |
| Vanilla_r4 | 1.3841 |
| MapPoPE-Flat_r4 | 1.3786 |

> Read the PRIMARY pair for the hierarchy question -- it is the only fully
> controlled comparison here. Hier-vs-flat carries a -0.96% param gap, and for
> the PoPE family also a rank gap (MapPoPE-Hier cannot accept bottleneck_r).
> Prior expectation from enwik8: the plain Hourglass scaffold was slightly WORSE
> on bpc (1.4844 vs 1.4727) at -18.75% FLOPs -- efficiency, not quality.
