# PAPER2X2 results (PAPER2X2_PREREG.md)

Torus paper task, held-out map (env-seed 10000), 300 ep cosine lr 1e-3, n=8 per arm, one batch.

## Per arm

| arm | T=128 | T=512 | T=1024 | final loss mean (range) |
|---|---|---|---|---|
| `RoPE` | 0.805 +/- 0.012 | 0.470 +/- 0.093 | 0.374 +/- 0.135 | 0.7701 (0.6762-0.8314) |
| `PoPE-Flat` | 0.679 +/- 0.058 | 0.639 +/- 0.046 | 0.607 +/- 0.033 | 0.8150 (0.7279-0.9573) |
| `Vanilla` | 0.971 +/- 0.043 | 0.915 +/- 0.052 | 0.777 +/- 0.094 | 0.0886 (0.0023-0.3839) |
| `MapPoPE-Flat` | 0.999 +/- 0.001 | 0.974 +/- 0.011 | 0.921 +/- 0.023 | 0.0063 (0.0000-0.0229) |
| `Vanilla_r4` | 1.000 +/- 0.000 | 0.985 +/- 0.006 | 0.928 +/- 0.012 | 0.0002 (0.0000-0.0003) |
| `MapPoPE_r4` | 1.000 +/- 0.001 | 0.982 +/- 0.018 | 0.941 +/- 0.028 | 0.0002 (0.0001-0.0005) |

## T=128

rule 9: r(final loss, acc) = -0.961 over 48 runs; acc = 1.002 -0.331*loss, resid sd 0.035

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=2 position main effect, raw | +0.243 | 0.038 | 0.038 | 8/8 | DETECTABLE |
| r=2 encoding main effect, raw | -0.049 | 0.038 | 0.038 | 1/8 | DETECTABLE |
| r=2 interaction, raw | +0.154 | 0.076 | 0.076 | 8/8 | DETECTABLE |
| r=2 position main effect, loss-matched | -0.004 | 0.013 | 0.013 | 3/8 | unmeasured |
| r=2 encoding main effect, loss-matched | -0.055 | 0.018 | 0.017 | 0/8 | DETECTABLE |
| r=2 interaction, loss-matched | +0.112 | 0.045 | 0.044 | 8/8 | DETECTABLE |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=4 position main effect, raw | +0.258 | 0.027 | 0.027 | 8/8 | DETECTABLE |
| r=4 encoding main effect, raw | -0.063 | 0.031 | 0.031 | 0/8 | DETECTABLE |
| r=4 interaction, raw | +0.125 | 0.063 | 0.062 | 8/8 | DETECTABLE |
| r=4 position main effect, loss-matched | -0.004 | 0.015 | 0.015 | 3/8 | unmeasured |
| r=4 encoding main effect, loss-matched | -0.055 | 0.019 | 0.019 | 0/8 | DETECTABLE |
| r=4 interaction, loss-matched | +0.110 | 0.039 | 0.038 | 8/8 | DETECTABLE |


## T=512

rule 9: r(final loss, acc) = -0.940 over 48 runs; acc = 0.974 -0.523*loss, resid sd 0.071

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=2 position main effect, raw | +0.390 | 0.079 | 0.079 | 8/8 | DETECTABLE |
| r=2 encoding main effect, raw | +0.114 | 0.030 | 0.030 | 8/8 | DETECTABLE |
| r=2 interaction, raw | -0.111 | 0.116 | 0.115 | 1/8 | unmeasured |
| r=2 position main effect, loss-matched | -0.000 | 0.049 | 0.048 | 3/8 | unmeasured |
| r=2 encoding main effect, loss-matched | +0.104 | 0.053 | 0.053 | 8/8 | DETECTABLE |
| r=2 interaction, loss-matched | -0.178 | 0.100 | 0.099 | 0/8 | DETECTABLE |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=4 position main effect, raw | +0.429 | 0.062 | 0.061 | 8/8 | DETECTABLE |
| r=4 encoding main effect, raw | +0.083 | 0.040 | 0.039 | 8/8 | DETECTABLE |
| r=4 interaction, raw | -0.172 | 0.075 | 0.074 | 0/8 | DETECTABLE |
| r=4 position main effect, loss-matched | +0.014 | 0.052 | 0.051 | 4/8 | unmeasured |
| r=4 encoding main effect, loss-matched | +0.095 | 0.049 | 0.049 | 8/8 | DETECTABLE |
| r=4 interaction, loss-matched | -0.196 | 0.097 | 0.096 | 0/8 | DETECTABLE |


## T=1024

rule 9: r(final loss, acc) = -0.875 over 48 runs; acc = 0.903 -0.517*loss, resid sd 0.107

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=2 position main effect, raw | +0.359 | 0.107 | 0.106 | 8/8 | DETECTABLE |
| r=2 encoding main effect, raw | +0.189 | 0.061 | 0.061 | 8/8 | DETECTABLE |
| r=2 interaction, raw | -0.089 | 0.188 | 0.186 | 4/8 | unmeasured |
| r=2 position main effect, loss-matched | -0.027 | 0.086 | 0.085 | 3/8 | unmeasured |
| r=2 encoding main effect, loss-matched | +0.179 | 0.083 | 0.082 | 8/8 | DETECTABLE |
| r=2 interaction, loss-matched | -0.154 | 0.177 | 0.175 | 1/8 | unmeasured |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| r=4 position main effect, raw | +0.444 | 0.075 | 0.074 | 8/8 | DETECTABLE |
| r=4 encoding main effect, raw | +0.123 | 0.062 | 0.062 | 8/8 | DETECTABLE |
| r=4 interaction, raw | -0.221 | 0.118 | 0.117 | 0/8 | DETECTABLE |
| r=4 position main effect, loss-matched | +0.034 | 0.072 | 0.071 | 5/8 | unmeasured |
| r=4 encoding main effect, loss-matched | +0.134 | 0.073 | 0.072 | 8/8 | DETECTABLE |
| r=4 interaction, loss-matched | -0.244 | 0.144 | 0.143 | 0/8 | DETECTABLE |

## Registered verdicts

- **H1** position main effect, r=2, T=128, raw: +0.243 (MDE 0.038) -> **STANDS (>= +0.15)**
- **H2** grows with length: T=128 +0.243 -> T=1024 +0.359 -> **MET**
- **H3** encoding main effect small: -0.049 (MDE 0.038) -> **MET**

## Determinism check against runs/sign (not replication, rule 27)

```
RoPE s0: compared 22, differing 0, losses equal True
Vanilla_r4 s0: compared 24, differing 0, losses equal True
```

## Reading (written after the analysis; not a registered verdict)

- **The headline survives the recipe, at about half its paper-recipe size at training length.**
  Converged, the r=2 position main effect at T=128 is +0.243 (MDE 0.038, 8/8), against +0.461
  under the paper's 16-epoch recipe. Index RoPE rises from 0.530 to 0.805. The effect grows beyond
  training length: r=2 +0.390 / +0.359 at T=512 / 1024; r=4 +0.429 / +0.444. All are detectable,
  8/8.
- **The encoding is not negligible once both index arms are off the floor.** It is slightly
  negative at training length (r=2 -0.049, r=4 -0.063; PoPE-Flat 0.679 against RoPE 0.805) and
  detectably positive beyond it (r=2 +0.114 / +0.189; r=4 +0.083 / +0.123). The position effect is
  the larger of the two at every length and rank. H3's registered form ("not detectably above 0.05
  at r=2, T=128") holds by 0.001, and it describes training length only.
- **PoPE helps the index code more than the path-integrated one beyond training length.** At r=4
  the interaction is detectably negative: -0.172 at T=512 and -0.221 at T=1024. At r=2 it is
  unmeasured at T=512 and T=1024, and positive at T=128 (+0.154).
- **The loss-matched position effect is uninformative here, not null.** Final losses of the two
  groups do not overlap: index arms 0.68-0.96, path-integrated arms 0.0000-0.38. The pooled
  acc ~ loss fit can equate the groups only by extrapolating across a loss interval (0.38-0.68)
  that no run occupies, and loss is itself a consequence of the position code. The residual
  therefore cannot separate "no position effect at matched loss" from "the position code causes
  the loss gap". This is related to MONOTONE's Q1, but it is not the same degeneracy: here the
  path-integrated group has within-group loss spread. `SIGN_ABLATION.md`'s loss-matched +0.123 /
  +0.195 is a different contrast (Signed_r4 - RoPE) in a pool that includes intermediate-loss
  monotone arms. The two loss-matched readings should not be compared. The RAW contrast is the
  registered primary.
- **Same runs as the sign batch for two arms.** RoPE and Vanilla_r4 seeds 0-7 are bitwise
  identical to `runs/sign/p0`, and their per-seed accuracies match exactly at all three lengths.
  For those arms this batch is not independent of `SIGN_ABLATION.md`.
- **Rank.** Both r=4 path-integrated arms converge on every seed (final loss <= 0.0005) and sit
  at 1.000 at training length. At r=2, Vanilla's final loss reaches 0.38 on its worst seed.
