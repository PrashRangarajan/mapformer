# INDIRECT_PREREG -- path integration on the PoPE paper's Indirect Indexing diagnostic

Written after a 1000-step smoke test of all four arms, before any full run.

## The paper's result (the replication target)

PoPE (Gopalakrishnan et al., arXiv:2509.10534 v3), Table 1, final-token accuracy on the test split,
3 seeds: **RoPE 11.16 +/- 2.45, PoPE 94.82 +/- 2.91**. Their reading (sec 5.1): RoPE entangles
"what" and "where", so it cannot index purely by position; PoPE decouples magnitude (content) from
phase (position) and solves the task.

## Task, as specified (App B.1)

Source string of 20-40 letters sampled without replacement from [A-Za-z]; a source character drawn
from it; a shift uniform in [-15, +15]; the target is the character that many places left or right.
Format `<source string>, <source char>, <shift>, <target char>`, character-level tokenisation,
splits 1M/10k/10k, cross-entropy and accuracy on the final token only. Chance = 1/52 = 0.019.

Recipe (App B.2/B.3): d_model 512, 8 heads, 8 layers, dropout 0, base wavelength 10,000, delta
init range 2pi; batch 64, lr 2e-4 cosine to 2e-5, weight decay 0.01, grad clip 1.0, AdamW
beta2 0.99, 100,000 iterations, 4,000 warmup.

**Deviations** (unavoidable or unstated): LayerNorm rather than RMSNorm (the repo's layers); block
size 56, since their stated sequence length of 40 cannot hold a 40-character string plus the
", c, -15, " suffix; targets that would fall outside the string are resampled; the path-integration
arms need an omega base, which the paper does not have -- set to the block size, 56. The PoPE
implementation is this repo's (`model_pope.py`), not the authors' code.

## The 2x2 this adds

The paper varies encoding only. This crosses it with position:

| | RoPE encoding | PoPE encoding |
|---|---|---|
| index position | RoPE (paper: 11.16) | PoPE (paper: 94.82) |
| path integration | MapWM | MapPoPE |

Path integration = position is a cumulative sum of learned per-token increments (MapFormer,
arXiv:2511.19279) rather than the token's index. r=2, the MapFormer default.

3 seeds per arm, one batch, all four arms trained together.

## Registered verdicts

- **R1 (replication)** RoPE < 0.30 and PoPE > 0.80 on the test split. If this fails, nothing else
  in the run is interpretable and the deviations above are the first suspects.
- **R2 (does path integration help the RoPE row?)** MapWM - RoPE, paired by seed. DETECTABLE if
  |delta| > MDE = 2.8 sd / sqrt(3). Direction registered: POSITIVE -- the task is pointer
  arithmetic, and a content-dependent increment is the mechanism that can express "move by k".
- **R3 (does it help the PoPE row?)** MapPoPE - PoPE. Registered prediction: no gain, because PoPE
  already solves the task -- a ceiling. Report as a ceiling cell if PoPE > 0.90.
- **R4 (interaction)** (MapPoPE - PoPE) - (MapWM - RoPE), reported with its MDE.
- **Floors**: chance 0.019; "copy the source character" (ignore the shift) and "copy a random
  character from the string" are computed in the analysis, since a model can score above chance
  without doing pointer arithmetic at all.
- **Convergence (rule 10)**: final-10% loss slope per 1k steps and the validation curve at every
  5,000 steps are recorded for each run.

n=3 gives an MDE of 1.6 sd, so only large effects are detectable; anything smaller is reported as
unmeasured (rule 11). Seeds follow the paper's count.

## Amendment 1 (2026-09-15, after the 3-seed batch)

The 3-seed batch shows the task is BIMODAL, not graded: a run either undergoes a late transition
(MapPoPE seed 0 lifts off at ~40k steps and ends at 0.981; PoPE seed 2 lifts off at ~65k and is
still rising at the budget end, 0.803) or sits flat at ~0.09 for all 100,000 steps. Seed means and
their MDEs are therefore the wrong statistic. Registered addition: seeds 3-7 at the same budget for
all four arms (8 total), and the primary statistic becomes the **solve rate** (fraction of seeds
above 0.5), compared against the paper's implicit 3/3 for PoPE and 0/3 for RoPE. No change to the
recipe. Also registered: PoPE seed 2's curve was still climbing at 100k, so the budget is a live
suspect for the replication failure and is reported as such, not as evidence against PoPE.

## Amendment 2 (2026-09-16) -- the budget test

PoPE's one solving seed was still climbing at the 100,000-step budget end (0.803), so the budget is a
live explanation for our 1/8 solve rate against the paper's implicit 3/3. Registered: rerun **PoPE
and MapPoPE at 200,000 iterations**, 8 seeds each, everything else identical. The cosine schedule is
stretched over the new budget (warmup stays 4,000), so this is a budget extension, not the same run
continued -- the learning-rate trajectory differs and that is the intended manipulation.

- **R5**: PoPE's solve rate at 200k. If it rises to >= 6/8 the replication failure was the budget; if
  it stays <= 2/8 the budget is exonerated and the implementation differences (delta clamp, LayerNorm
  vs RMSNorm) become the remaining suspects.
- **R6**: MapPoPE's solve rate at 200k, against its 5/8 at 100k. Registered prediction: it rises or
  holds; a drop would mean the 100k result was schedule-specific.
- **R7**: the MapPoPE - PoPE solve-rate gap at 200k (Fisher exact), against 5/8 vs 1/8 (p = 0.119) at
  100k. This is the contrast the amendment exists to power.
- Also recorded per run: the step at which validation accuracy first exceeds 0.5 (lift-off step), so
  "solves later" is separable from "never solves".
- RoPE and MapWM are NOT rerun at 200k: neither has a seed above 0.09 and neither shows any upward
  trend. Cross-budget comparisons against them are therefore unmatched and are not made; the
  registered contrasts above are all within the 200k batch.
