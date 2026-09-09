# EM vs WM x rank on allocentric MiniGrid-DoorKey-16x16

5 arms x **8 seeds**, ONE batch, egocentric observation, allocentric action recoding,
50 epochs, 25K cached buffer, no `--fast-attn` (invalid for MapEM). Parameter spread
across arms **under 0.11%**. Pre-registration: `MINIGRID_EM_PREREG.md`.
Measured floor: T=128 **0.635**, T=512 **0.536**, T=1024 **0.490**.

| arm | position | T=512 | T=1024 | sd @1024 | **min @1024** |
|---|---|---|---|---|---|
| `RoPE` | index | 0.819 ± 0.007 | 0.788 ± 0.019 | 0.019 | 0.754 |
| `Vanilla` (WM, r=2) | path-int | 0.817 ± 0.021 | 0.778 ± 0.045 | 0.045 | 0.693 |
| **`Vanilla_r4`** (WM, r=4) | path-int | **0.843 ± 0.011** | **0.822 ± 0.015** | **0.015** | **0.798** |
| `VanillaEM` (EM, r=2) | path-int | 0.797 ± 0.060 | 0.763 ± 0.081 | 0.081 | 0.572 |
| `VanillaEM_r4` (EM, r=4) | path-int | 0.826 ± 0.046 | 0.803 ± 0.056 | 0.056 | 0.666 |

## P1 CONFIRMED, and it needs BOTH fixes

| contrast @ T=1024 | delta | MDE | seeds + | verdict |
|---|---|---|---|---|
| `Vanilla` (r=2) - `RoPE` | -0.010 | 0.056 | 4/8 | unmeasured |
| **`Vanilla_r4` - `RoPE`** | **+0.034** | 0.014 | **8/8** | **DETECTABLE** |
| `VanillaEM` (r=2) - `RoPE` | -0.026 | 0.087 | 3/8 | unmeasured |
| `VanillaEM_r4` - `RoPE` | +0.015 | 0.062 | 7/8 | unmeasured |

**Path integration beats an index code on this benchmark -- but only with allocentric
recoding AND `r=4`.** At `r=2` it is indistinguishable from index (-0.010, 4/8),
which is the configuration every prior MiniGrid result used. Two cheap fixes, neither
architectural, and both are needed.

## P2 REFUTED: EM never beats WM here

| contrast | T=512 | T=1024 |
|---|---|---|
| EM - WM at r=2 | -0.020 (3/8) | -0.016 (4/8) |
| EM - WM at r=4 | -0.018 (2/8) | -0.019 (4/8) |

All four unmeasured, all four **negative**. The prediction was that EM's AND-gate
would filter noisy egocentric content, as it appears to on DoorKey BC. It does not.

## P3 unmeasured: rank does NOT help EM more

Rank buys WM `+0.044` and EM `+0.041` at T=1024 -- interaction `-0.003` against an
MDE of `0.128`. A better-conditioned phase helps both equally; it does not
preferentially repair EM's collapsing `A_P`.

## P4 CONFIRMED, and it is the actual finding

**EM's entire mean deficit is one seed.** Drop each arm's single worst seed --
symmetrically, so no arm is favoured:

| | mean | drop each arm's worst | EM - WM after |
|---|---|---|---|
| r=2 | WM 0.778, EM 0.763 (**-0.016**) | WM 0.790, EM 0.790 | **-0.001** |
| r=4 | WM 0.822, EM 0.803 (**-0.019**) | WM 0.825, EM 0.823 | **-0.003** |

And the failures are **collapses, not degradations** -- the gap between an arm's
worst and second-worst seed:

| | worst -> 2nd worst | gap |
|---|---|---|
| WM r=4 | 0.798 -> 0.805 | **0.007** |
| EM r=4 | 0.666 -> 0.810 | **0.144** |
| WM r=2 | 0.693 -> 0.724 | 0.031 |
| EM r=2 | 0.572 -> 0.739 | **0.167** |

`VanillaEM_r4`'s per-seed scores are **0.666, 0.810, 0.813, 0.818, 0.825, 0.828,
0.830, 0.834** -- seven seeds tightly bunched at WM_r4's level, and one 0.14 below
the pack. WM's distribution has no such gap at either rank.

**So EM is not worse than WM on this task. EM MATCHES WM and fails catastrophically
on 1 seed in 8.** That is exactly what a rank-one AND-gate with no additive fallback
predicts, it was pre-registered as the expected shape, and a mean alone reports it as
"EM is worse" -- which is how the `n_obs=256` vocab result was previously read.

## What this changes

- **`Vanilla_r4` is the arm to use on this benchmark.** Best mean (0.822) AND the
  tightest spread (sd 0.015, min 0.798) of any arm measured, index or otherwise.
- **The published MiniGrid conclusion needs its scope narrowed.** "Index codes win on
  MiniGrid" was measured without allocentric recoding and at r=2. With both, path
  integration wins 8/8.
- **EM's problem here is reliability, not capability.** The open question is whether
  the collapse is fixable by initialisation -- `Level15EM` needed
  `log_R_init_bias=3.0` for exactly this reason, and shared `p_0` is worth +0.358
  elsewhere. Neither was applied here.

## Caveats

n=8 with EM sd up to 0.081 gives an MDE near 0.10 on EM contrasts -- only the
`Vanilla_r4 - RoPE` comparison (sd 0.015) is well powered. Headroom above the best
arm is ~0.18 to a 1.0 ceiling, so this is not a ceiling artefact. One environment,
one tokenization, flat models only (no hierarchy, no PoPE).
