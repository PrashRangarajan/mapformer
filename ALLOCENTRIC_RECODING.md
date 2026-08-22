# Allocentric action recoding fully restores MapFormer under rotation actions

`KNOB_SWEEP.md` established that rotation-based actions (turn-left / turn-right /
forward) cut the position effect from +0.478 to +0.049 — twice the next largest
knob and 90% of the total swing between the torus and MiniGrid. Proposed
mechanism: MapFormer path-integrates by cumsumming a **fixed per-token delta**,
and under turn/forward the displacement depends on the accumulated heading, which
that form cannot represent.

This tests it by changing **only what the token stream records**. Dynamics are
identical — same trajectories, same observations, and the answer-stream gates come
out the same to three decimals (o1 0.501 / o2 0.472 / o3 0.440 / o5 0.462 against
a 0.507 marginal, both ways). Under `--action-record allocentric` the recorded
token is the absolute displacement that actually occurred, or STAY when the step
produced none.

n=3, matched budget (392 batches), floor measured at 0.508.

| condition | Vanilla | RoPE (index) | position effect | paired |
|---|---|---|---|---|
| rotate, commanded (turn/forward) | 0.557 ± 0.023 | 0.508 ± 0.007 | **+0.049** | +0.063/+0.053/+0.031 |
| **rotate, allocentric** | **0.994 ± 0.008** | 0.508 ± 0.007 | **+0.485** | +0.483/+0.490/+0.484 |
| *baseline (translate actions)* | *0.989 ± 0.010* | *0.511 ± 0.012* | *+0.478* | — |

## The recovery is complete, not partial

MapFormer goes from **0.557 to 0.994** — above the 0.989 it reaches on the
ordinary translate baseline — and the position effect goes from +0.049 to
**+0.485**, against baseline's +0.478. Recoding the action stream recovers
**100% of the lost effect**, on 3/3 seeds with a spread of ±0.008 and paired
differences of +0.483/+0.490/+0.484.

**So rotate's collapse was entirely a representation mismatch, not task
difficulty.** The environment was never harder: same walk, same map, same
questions. What changed is whether the action token names something the
path integrator can accumulate.

Note also that the index model is **unmoved at exactly the floor** (0.508 both
ways). It learns nothing from either encoding, so the recovery is specific to the
path-integrating architecture rather than a general easing of the task.

## The mechanism claim is confirmed, and it comes with a fix

MapFormer's `ActionToLieAlgebra` maps each token to a fixed element of the Lie
algebra and `PathIntegrator` cumsums it. That is well-specified exactly when the
action token determines the displacement. Turn/forward violates this; absolute
displacements satisfy it.

**The remedy is available in any setting where the agent's heading is known** —
which is every simulator, including MiniGrid and Habitat, since the pose is part
of the state. Recode the action stream before tokenising; nothing about the
architecture needs to change.

## Why this matters beyond the torus

Rotation-based action spaces are the norm in embodied navigation, not the
exception: MiniGrid, MiniWorld, Memory Maze and Habitat all use
turn-left / turn-right / move-forward. `MINIGRID_2X2X2.md` measures MapFormer-WM
losing to an index baseline on DoorKey-16x16 at long horizon (0.792 vs 0.860),
and this identifies why and what to do about it.

It also sharpens what the paper's own Appendix B.2.2 gestures at. That section
motivates non-commutative action groups with the mother/father example and then
validates on synthetic 4D rotations. Turn/forward navigation is the everyday case
its argument predicts should fail — and it does, by 0.429 — but the resolution
here is not a richer group. It is to record the actions in a frame where the
existing commutative machinery already applies.

## Limits

- **n=3, one environment, one budget.** The effect is enormous and tight
  (±0.008), but it is three seeds.
- **STAY is an extra token.** Allocentric recording adds one symbol to the
  vocabulary (21 -> 22). The comparison is therefore not exactly
  parameter-matched at the embedding layer; the difference is one row of a 128-d
  embedding, against an effect of +0.436.
- **Not tested where displacement is continuous.** In Habitat, "forward" moves a
  real-valued distance, so the allocentric recoding would need to emit a
  quantised or continuous displacement rather than one of four symbols. Whether
  the recovery survives that is untested.
