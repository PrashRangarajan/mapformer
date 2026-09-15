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
