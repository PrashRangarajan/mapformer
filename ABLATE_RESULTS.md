# PoPE component ablation: the bound account is dead, and most of Table 5 replicates after all

Pre-registration `ABLATE_PREREG.md`. Code corpus, 4 arms x 3 seeds in one batch,
trained at 512, evaluated at 512 and 2048. `PoPE-Full` was verified to reproduce
`PoPE-Flat` to maxdiff 0.00e+00 before launch.

> **CORRECTED 2026-09-22.** The first version of this file said "their Table 5
> DOES NOT REPLICATE" and "removing softplus entirely is free". Both were wrong,
> for two separate reasons, and are corrected in place below. An audit against
> the authors' released code also verified our PoPE implementation as faithful
> (max logit difference **1.7e-06** on identical inputs, including the frequency
> ladder, the delta clamp and side, and the score expansion).

## Two errors in the first version

**1. A contaminated checkpoint.** `PoPE-NoSigma_s2.best.pt` held **iteration
15000 at val 0.9962** while the run's own JSON records `best_val_bpc 0.9156` at
36000. A duplicate launch of that arm (see the driver race below) wrote its own
early best over the finished run's checkpoint **55 minutes after the run
completed**. `eval_code_long.py` could not catch it: the architecture matched, so
the load succeeded. That single file produced the "anomalous seed" (0.9647) and
the "plausible instability from unbounded magnitudes" caveat. **Both are void.**
Everything below is re-scored on `.final.pt`. The eval now refuses any checkpoint
whose stored `val_bpc` disagrees with its run's JSON.

**2. The in-distribution readout was too noisy to support the claim.** The
trainer's val uses 40 random windows -- about 5% of the val file. Recomputed over
the **whole** val file the ordering changes, and NoSigma's sign flips.

## A3 -- in distribution, full val file at seq 512 (`.final.pt`)

| arm | bpc | vs Full | relative | MDE | verdict | theirs, 124M |
|---|---|---|---|---|---|---|
| **PoPE-Full** | **0.87930** | -- | -- | -- | -- | 21.33 (best) |
| PoPE-NoDelta | 0.88035 | +0.00104 | +0.119% | 0.00113 | unmeasured, 3/3 worse | 21.42 (+0.138%) |
| PoPE-NoSigma | 0.88043 | +0.00113 | +0.129% | 0.00458 | **unmeasured** | 21.57 (+0.366%) |
| PoPE-ReLU | 0.89117 | +0.01187 | **+1.350%** | 0.00127 | **DETECTABLE**, 3/3 | 21.55 (+0.335%) |

**Ours: Full < NoDelta ~ NoSigma < ReLU. Theirs: Full < NoDelta < ReLU ~ NoSigma.**
Full best and NoDelta second **replicate**, and NoDelta's size is close to theirs
(+0.119% vs +0.138%). NoSigma's sign now agrees. **Only ReLU's rank differs**,
and it differs because ReLU is worse for us, not because sigma is free.

**The NoSigma cell is UNMEASURED, not refuted.** Our MDE there is **0.388%
relative** against their 124M effect of **0.366%** -- we could not have detected
their effect if it were present. Rule 11: this is not a non-replication.

## A1/A2 -- out of distribution (eval at 2048, `.final.pt`)

| arm | 0-512 | 512-1024 | 1024-2048 | extrapolation penalty |
|---|---|---|---|---|
| PoPE-Full | 0.8763 | 0.7775 | 0.8388 | -0.0375 |
| **PoPE-NoDelta** | 0.8774 | 0.7756 | **0.8326** | -0.0448 |
| **PoPE-NoSigma** | 0.8776 | 0.7705 | 0.8668 | **-0.0108** |
| PoPE-ReLU | 0.8885 | 0.7948 | 0.8822 | -0.0063 |
| *RoPE (runs/code)* | 0.8755 | 2.6713 | 4.4641 | **+3.5885** |

- NoSigma - Full: +0.0281 (MDE 0.0509, 0/3) unmeasured; penalty +0.0267 unmeasured
- ReLU - Full: +0.0435 (MDE 0.0570, 0/3) unmeasured; penalty +0.0312 unmeasured
- **NoDelta - Full: -0.0061 (MDE 0.0033, 3/3) DETECTABLE** -- removing delta is
  slightly BETTER out of distribution; penalty -0.0073 (MDE 0.0044) DETECTABLE

**F2 FIRES, as registered. The non-negativity account is dead.** It held that
PoPE's magnitudes are non-negative, so `|score(D)| <= score(0)` at every offset,
and that this is why PoPE-encoding arms survive past their context while
RoPE-encoding arms blow up. **NoSigma has signed magnitudes exactly like RoPE and
does not blow up** (-0.0108 against RoPE's +3.5885). Correcting the stale
checkpoint moved NoSigma's penalty from -0.0234 to -0.0108 and did not threaten
this: the contaminant was undertrained, so the correction can only help NoSigma.

Also refuted by direct measurement (audit): identity magnitudes do not
destabilise anything. NoSigma's logit sd is 2.50 against Full's 3.34, and it
rebuilds an equivalent recency kernel with a mean magnitude near zero by training
a smaller `Wq`. There is no pathology for the softplus to be preventing.

## Why the fourth cell differs: there is no gain to decompose

Their Table 4 gives RoPE 21.55 and PoPE 21.33 at 124M, so Table 5 measures how
much of a **0.22-ppl PoPE-over-RoPE gain** each component carries. On this corpus
that gain does not exist:

    PoPE - RoPE (runs/code, same batch, n=3): +0.0034 bpc, MDE 0.0101, 1/3 better
    -- unmeasured, and the point estimate is REVERSED: PoPE is WORSE.

That single fact predicts NoSigma ~ Full without appealing to scale or
implementation, and it is the ranked-first explanation.

## Three findings that stand on their own

- **delta is nearly inert here.** **80.7%** of `pope_delta` parameters sit at or
  above 0, inside the `clamp(-2pi, 0)` dead zone where the gradient is exactly
  zero, and freeze permanently; the largest effective |delta| across all seeds is
  0.60 rad, **9.6%** of its range. Their code has the same clamp and the same zero
  init, but at 100k steps and lr 6e-4 the live fraction has far more travel. Our
  Full-vs-NoDelta contrast is therefore a sub-radian perturbation and should be
  read as one.
- **ReLU kills about half the magnitude channels** (audit: 50.5% exactly dead).
  Losing half of a 64-channel frequency ladder costs more at 28.6M than at 124M,
  which is the likeliest reason ReLU is 4x worse for us than for them. This is an
  injury orthogonal to the PoPE-vs-RoPE axis.
- **Batch-to-batch reproducibility floor: mean |delta| 0.0021, max 0.0028 bpc**
  for the same architecture at the same seeds in two different batches
  (`runs/code/PoPE-Flat` vs `runs/code_ablate/PoPE-Full`). **That floor exceeds
  the NoDelta effect** (+0.00104) and is comparable to the NoSigma MDE.

## Caveats

- **Nothing is converged.** Every arm is still falling at -0.0021 to -0.0034 bpc
  per 1000 iterations at 36k, under a constant LR with no warmup, no decay and no
  weight decay. Their recipe is warmup + cosine to lr/10 with AdamW wd 0.01.
- Scale: 28.6M against their smallest at 124M, and their own sigma effect nearly
  DOUBLES from 124M to 253M.
- **"Without sigma()" is our inference.** The paper never defines it, and the
  authors' released code has no sigma switch at all -- softplus is hardcoded, so
  their Table 5 sigma rows are not reproducible from their own release. Identity
  is the natural reading given a separate ReLU row, but mu=|q| or mu=1 are not
  excluded.
- n=3 out of distribution: this establishes "none of the PoPE variants break",
  not an ordering among them.

## What would settle the remaining ambiguity, cheapest first

1. **~4 GPU-h, n=1.** Train the two other readings of "without sigma()" (mu=|q|,
   mu=1). If either reproduces a ~0.37%-relative gap where identity does not, the
   whole question is an interpretation mismatch. Do this before spending on seeds.
2. **~32 GPU-h.** `PoPE-Full`, `PoPE-NoSigma` and `RoPE`, n=8, under the paper's
   recipe (warmup + cosine, AdamW wd 0.01, dropout 0, clip 1.0). MDE ~0.34%
   relative, just enough for their 124M effect -- and the RoPE arm says whether
   PoPE wins at all under that recipe. If it still does not, explanation 1 is
   confirmed and nothing further is warranted.
