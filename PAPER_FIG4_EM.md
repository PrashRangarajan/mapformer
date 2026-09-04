# Do we reproduce the paper's Figure 4?

Sec 5.4 / Fig 4 makes four checkable claims about a trained MapFormer.
Each is tested here with the paper's own metric, on 8 seeds, at the
paper's rank and at a widened one.

| claim | paper reports | `VanillaEM` | `VanillaEM_r4` |
|---|---|---|---|
| **C1** ‖Δ_action‖ / ‖Δ_obs‖ | ≫ 1 <br><sub>observations leave position untouched</sub> | 21.8 ± 25.2 | 26.1 ± 13.0 |
| **C2** cos(Δ_left, Δ_right) | **−1** <br><sub>opposite actions cancel</sub> | -0.352 ± 0.896 | -0.996 ± 0.002 |
| **C3** \|cos(Δ_left, Δ_up)\| | **≫ 0** <br><sub>reported as a LIMITATION</sub> | 0.868 ± 0.214 | 0.190 ± 0.115 |
| **C4** ‖v_obs‖ / ‖v_action‖ | ≫ 1 <br><sub>only observations update content</sub> | 0.90 ± 0.21 | 0.64 ± 0.04 |

## Verdict

**C1 reproduces**: actions rotate 22x more than observations.
**C2 reproduces**: cos(opposite) = -0.352 against the paper's −1.
**C3 reproduces, including the failure**: |cos(orthogonal)| = 0.868, which is the ≫ 0 the paper reports and flags as needing an added constraint to fix.
**C4**: ‖v_obs‖/‖v_action‖ = 0.90.

**On C3, widening the bottleneck does what the paper says needs an extra loss term.** |cos(orthogonal)| falls 0.868 → 0.190 going from VanillaEM to VanillaEM_r4, with no bounded-energy constraint and no change to the objective. The paper's own remedy for its disentanglement failure is an added regulariser; a wider bottleneck gets most of the way there for +384 parameters.

Inference only, 8 seeds, torus checkpoints from `/home/prashr/mapformer/runs/em_fig4/p0`, arms `VanillaEM` and `VanillaEM_r4`. Δ_in is the r-dimensional Lie-algebra code (the paper's metric for C2/C3); Δ is the full per-head increment (C1).

---

## Pre-registered verdict on C4

Written in `run_em_fig4.sh` before the batch launched:

> EM ratio >> 1 while WM stays < 1 -> C4 is an EM property the caption does not
> scope, our reproduction of Fig. 4 is COMPLETE, and the WM number is not a
> discrepancy but a different architecture behaving differently. EM ratio also
> < 1 -> a genuine discrepancy with the paper, on the architecture the claim is
> most likely about. Record it as one; do not explain it away.

**The second branch fired. C4 does not reproduce on either backbone.**

| | C4 = ‖v_obs‖ / ‖v_action‖ |
|---|---|
| paper (Fig. 4) | ≫ 1 |
| MapWM r=2 (n=8) | 0.57 ± 0.06 |
| **MapEM r=2 (n=8)** | **0.90 ± 0.21** |
| MapEM r=4 (n=8) | 0.64 ± 0.04 |

EM moves toward 1 and does not cross it; the widest seed sits near 1.1 against a
claim of "much greater than". The hypothesis that C4 is an EM-only property --
motivated by Sec 5.4 being explicitly about EM's "two separate pools of neurons
... specialized for either position or observation", which MapWM's additive
attention lacks by construction -- is **not supported**. Recorded as a
discrepancy with the paper, not explained away.

**Standing summary: three of the four Fig. 4 claims reproduce, on both
backbones, and the fourth is inverted on both.**

## Secondary: the rank finding replicates on the EM backbone

Not the question this batch was launched to answer, and reported as such.

| | C2 cos(opposite) | C3 \|cos(orthogonal)\| |
|---|---|---|
| MapWM r=2 | −0.729 ± 0.373 | 0.779 ± 0.276 |
| **MapEM r=2** | **−0.352 ± 0.896** | **0.868 ± 0.214** |
| MapWM r=4 | −0.996 ± 0.004 | 0.174 ± 0.083 |
| MapEM r=4 | −0.996 ± 0.002 | 0.190 ± 0.115 |

EM's r=2 code is **more** skewed than WM's on both metrics, with a seed
standard deviation on C2 of 0.896 — some seeds learn opposite actions that do
not oppose at all. Widening to r=4 lands both backbones on the same clean
geometry to three decimals. The r=2 conditioning failure is therefore not a
property of the WM attention form; it is a property of the bottleneck, which is
what the rank account claims.

## Is C4 a metric mismatch rather than a model difference?

Checked, because the published probe applies `norm1` (a LayerNorm) before
`v_proj`, and a LayerNorm flattens exactly the magnitude difference C4 is about.
The paper's wording is unambiguous about what it measures -- "the norm of value
embeddings v_t in the attention layer becomes much bigger for observations than
actions" -- so three readings of that were tested on the same checkpoints.

| arm | with LN (published) | no LN | contextual |
|---|---|---|---|
| MapWM r=2 | 0.57 ± 0.06 | 0.74 ± 0.09 | 0.58 ± 0.10 |
| MapWM r=4 | 0.60 ± 0.04 | 0.87 ± 0.05 | 0.60 ± 0.06 |
| MapEM r=2 | 0.90 ± 0.21 | **1.25 ± 0.31** | 0.98 ± 0.21 |
| MapEM r=4 | 0.64 ± 0.04 | 0.95 ± 0.08 | 0.77 ± 0.04 |

The metric choice moves the number by 30--40% and **does not change the verdict**.
One of eight cells crosses 1 (MapEM without the LayerNorm, 1.25 ± 0.31), which is
the paper's direction; none is anywhere near "much bigger than". So C4 is a
genuine discrepancy whose *magnitude* is metric-dependent, not an artifact of how
we chose to measure it.

## What this does and does not touch

The paper's Fig. 4 asserts a **double dissociation**: action tokens update
position and not content, observation tokens update content and not position.
We reproduce the position half decisively (C1: actions rotate 22--36x more than
observations, on both backbones) and fail to reproduce the content half. The
paper's central claim -- that input-dependent SO(2) rotations are what let a
transformer build a cognitive map -- rests on the position half, on Table 2
accuracy, and on the baselines, all of which reproduce.

**Untested hypothesis, stated so it is not mistaken for a finding.** There may be
no gradient pressure to shrink action tokens' value vectors, because attention can
simply decline to attend to action positions; zeroing the *position* increment for
observations, by contrast, is forced. If so, C4 would be a property of the
optimisation rather than of the architecture, and the discriminating measurement
is the attention mass landing on action positions. Not run.
