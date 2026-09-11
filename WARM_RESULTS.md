# Warm-start: EM can represent recency exactly -- and training dismantles the solution

> **CORRECTED 2026-09-11 by `UNFREEZE_RESULTS.md`.** W4's reading below -- "training
> dismantles the solution, a landscape property", with an early-window mechanism -- is
> WITHDRAWN in its strong form. Installing the identical Delta at 8x the weight scale
> raises the trainable twin from 0.642 to 0.941 (+0.298, 7/8), closing 84% of the gap:
> most of W4 was Adam eroding a rewind I stored at a 0.0156-per-step weight scale. The
> early-window mechanism is refuted outright -- released after the content branch has
> trained, the rewind breaks within 1-6 epochs. W1 (1.000 frozen) and "0 of 40 from-scratch
> runs find a rewind" are unaffected. The deficit is a SEARCH problem.

Pre-registration: `WARM_PREREG.md`. Single-`p0` EM with the constructed recency
rewind installed in its position pathway; content branch random. Seeds 0-7.
Comparators: `VanillaEM_P0_r4` (from scratch) and `Vanilla_r4` (WM), same seeds,
existing deterministic runs (determinism re-verified by the MagOnly batch).

| arm | acc T=1024 | acc T=2048 | final loss | worst seed |
|---|---|---|---|---|
| **`EMWarm_freeze`** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.007 | 1.000 |
| `Vanilla_r4` (WM) | 0.975 +/- 0.072 | -- | 0.064 | 0.797 |
| `EMWarm_train` | 0.642 +/- 0.266 | 0.582 +/- 0.249 | 1.216 | 0.172 |
| `VanillaEM_P0_r4` (from scratch) | 0.600 +/- 0.126 | -- | 1.340 | 0.365 |

## W1 CONFIRMED -- the full one-layer EM represents recency exactly

`EMWarm_freeze` scores **1.000 on 8/8 seeds, at the training length AND at 2x it**,
with per-offset accuracy 1.00 at every `k` from 1 to 64. Given the rewind, the
content branch learns the token-type gate and the readout on its own. The audit's
kernel-level existence proof is now a full-model demonstration, and single-`p0` EM
with the right position code does at least as well as WM on these seeds.

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| W2 `freeze - P0 from scratch` | **+0.400** | 0.126 | 0.125 | **8/8** | **DETECTABLE** |
| W3 `freeze - WM` | +0.025 | 0.072 | 0.071 | 1/8 | unmeasured (matches WM) |
| W4 `train - freeze` | **-0.358** | 0.266 | 0.263 | **0/8** | **DETECTABLE -- falsifier fires** |

## W4 REFUTED -- the solution is not held when training can reach it

The trainable twin starts from the SAME installed rewind and ends indistinguishable
from from-scratch EM (0.642 vs 0.600). Its position pathway no longer implements the
rewind: pooled slope -0.055 against -1, bulk block median -0.042, with remnants in
0-4 of 63 blocks; its position kernel picks the answer on 15% of queries, down from
100%. Per-offset accuracy decays with `k` exactly as from scratch (0.99 at k=1 to
0.55 at k=64).

**It never uses the rewind -- it is not learned and then lost.** The frozen twin is at
loss 0.19-0.46 by epoch 10 and ~0.03 by epoch 30. The trainable twin sits at chance
(~2.75) for ~100 epochs on every seed, then climbs late toward a from-scratch-like
solution. So the damage happens early, while the content branch is still random.

**Weight decay is excluded as the cause.** The frozen twin is exempt from decay on its
position pathway and the trainable twin is not -- a confound in W4 as designed. But
over the first 480 steps, mostly LR warmup, sum(lr) is ~0.16, so AdamW's decoupled
decay shrinks weights by a factor of 0.992. Under 1% cannot dismantle a rewind whose
loss consequence is visible by epoch 10. Gradient dynamics did it.

*Limit:* intermediate checkpoints were not saved, so "destroyed early" is inferred
from the loss trajectory, not measured on the weights at epoch 10.

## From scratch, the rewind is never found

`_REWIND_PROBE.json`: across all 40 from-scratch EM runs (5 parameterisations x 8
seeds), the pooled rewind slope is 0.000 +/- 0.02, and 1 of ~2,550 (head, block)
pairs has a slope below -0.5. This is robust to how the rewind is measured.

## What this establishes

The EM recency deficit (-0.375 against WM) is a property of the LOSS LANDSCAPE, and
it has three parts, each now measured:

1. **The solution exists in the architecture** (W1: 1.000, 8/8, both lengths).
2. **Training from scratch never finds it** (no rewind in any of 40 runs).
3. **Training does not hold it when given it** (W4: installed rewind dismantled
   before the content branch can use it).

Audit finding 2 is upgraded from kernel level to full model, and sharpened: this is
not only search difficulty, the correct position code is unstable under joint
training.

## A hypothesis this suggests, stated as one

EM's score is `A_X (*) A_P`, so `d score / d A_P = A_X`: the position pathway's
gradient is gated by the CONTENT scores. While the content branch is random, the
position pathway receives gradients filtered through a random gate -- noise that is
large enough to dismantle a correct rewind in the first few epochs. The frozen twin
works because freezing protects the position code through that window. This is the
EM half of the withdrawn Thm 3, restated where it is actually true (EM's product), and
it predicts that the damage is confined to the early window.

**Untested.** The direct test is a two-phase schedule: install the rewind, freeze the
position pathway for N epochs until the content branch has learned, then unfreeze.
If the rewind survives, EM's problem is purely the early window and has a curriculum
fix; if it is dismantled even with a trained content branch, the instability is
intrinsic to the landscape. Saving intermediate checkpoints would also turn the
"destroyed early" inference into a measurement.
