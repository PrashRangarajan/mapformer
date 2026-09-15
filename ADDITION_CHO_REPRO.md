# Reproducing Cho et al.'s position-coupling result before comparing anything against it

Written 2026-09-14, before training. The positive control in `ADDITION_PILOT2.md` did not generalise to 2x the
training length, so the MapFormer comparison was not read. This run reproduces the control under Cho et al.'s
recipe (arXiv:2405.20671, Appendix C, Table 1) first.

**Target.** Their 1-layer, 4-head model trained on 1-30-digit additions reaches 95.65% exact match at 200 digits
(median over 8 runs).

**Recipe, from their Table 1:**
- 1 layer, 4 heads, d = 512 (128 per head), feed-forward 2048;
- Adam, lr 1e-4, linear warmup over 1% of steps, cosine decay to 0.1 lr;
- 50,000 steps x batch 1,000;
- max_pos 202;
- training on 1-30 digits with balanced digit-length sampling;
- shared digit tokens, reversed sum, zero-padded operands.

**Two runs, seed 0:**

| run | architecture | what it isolates |
|---|---|---|
| `ChoCoupledAPE` | their block as read from Table 1 (`model_cho_coupled.py`): GEGLU, RMSNorm pre and post, no dropout | does their result reproduce here? |
| `CoupledAPE` | this repo's WM layer (GELU, LayerNorm, dropout 0.1) at d=512, same recipe | recipe vs architecture |

**Stated deviations:**
- training problems are sampled fresh every batch rather than drawn from a fixed 1M-example set;
- "PreNorm and PostNorm" is read as a sandwich block;
- the coupled start ID is at least 2, so no digit shares ID 0 with BOS/PAD;
- evaluation uses 256 problems per length (they use 100k).

**Pass condition, fixed now:** `ChoCoupledAPE` exact match >= 0.9 at 200 digits.
- If it passes, this recipe becomes the base for the MapFormer arms.
- If it fails, the gap to the published result must be understood before any addition comparison is run.

Evaluation lengths: 30, 60, 100, 150 and 200 digits.

## Results (seed 0, 256 problems per length, trained on 1-30 digits)

| run | final loss | 30 | 60 | 100 | 150 | 200 | per-digit at 200 |
|---|---|---|---|---|---|---|---|
| `ChoCoupledAPE` (their block) | 0.0000 | 1.000 | 1.000 | **0.938** | 0.590 | 0.023 | 0.979 |
| `CoupledAPE` (this repo's layer, same recipe) | 1.7546 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.109 |

(exact match by operand length)

## Verdict on the pass condition: NOT MET

`ChoCoupledAPE` reaches 0.023 exact match at 200 digits, not >= 0.9. But it generalises cleanly to 100 digits
(0.938, 3.3x the training length) and partly to 150 (0.590). That is far beyond pilot 2's control (1.5x). It
does not reach Cho et al.'s reported 95.65% at 200.

## What the two runs show

- **The architecture, not the recipe, was most of the pilot-2 gap for the oracle.** Their block at their
  recipe generalises to 100 digits. This repo's WM layer at the same recipe does not train at all: loss is
  still 1.75 after 50k steps, and its per-digit accuracy stays at chance. It did train at lr 1e-3 in pilot 2,
  so the repo layer and lr 1e-4 do not work together.
- **The remaining gap to 200 digits is unexplained,** with candidates untested:
  - one seed against their median of 8;
  - fresh sampling instead of their fixed 1M set;
  - the sandwich-norm reading;
  - how often the highest position-ID rows are trained;
  - 256 evaluation problems against their 100k.

  Per-digit accuracy at 200 is 0.979, so the model is close: 200 digits with 2% per-digit error rarely
  come out exactly right.

## Consequence for the MapFormer comparison

MapFormer's arms use the repo's WM layer, which does not train under this recipe. A fair comparison needs
every arm in the SAME block. The position mechanism (coupled APE, index RoPE, NoPE, signed or monotone
path-integrated rotation) would be swapped inside the Cho block, the recipe that works would be used, and a
readout length pre-registered where the control is known to pass: 100 digits, 0.938 here.
