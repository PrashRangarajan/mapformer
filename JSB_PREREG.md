# JSB_PREREG -- Bach Chorales (PoPE paper Table 2) with path integration

Written after a 100-iteration smoke run, before any full run. Queued behind the Indirect Indexing
batch so the two do not contend for GPUs.

## The paper's result (the replication target)

PoPE (arXiv:2509.10534 v3) Table 2, "Best NLL on the test split": **RoPE 0.5081, PoPE 0.4889**
(a gain of 0.0192). No seed count or spread is reported for this table.

## Protocol

Dataset exactly as specified: `Jsb16thSeparated.json` from github.com/czhuang/JSB-Chorales-dataset,
229/76/77 train/valid/test pieces (our copy matches), raster-scan serialisation (voices down, then
time across), silence as its own token, vocabulary 90, maximum sequence length 2048.
Recipe exactly as specified (App B.2/B.3): d_model 256, 8 heads, 6 layers, dropout 0.2, base
wavelength 10,000, delta init range 2pi; batch 4, lr 6e-4 cosine to 6e-5, weight decay 0.01, grad
clip 1.0, AdamW beta2 0.99, 3,000 iterations, 10 warmup.

**Deviations**: LayerNorm rather than RMSNorm (the repo's layers); the path-integration arms need an
omega base, which this paper has no analogue for -- set to 2048, the sequence length; PoPE is this
repo's implementation, not the authors' code; evaluation every 250 iterations, and "best NLL" is
read two ways (test at the best validation step, and best test directly, which is what the paper's
wording literally says).

## The 2x2

| | RoPE encoding | PoPE encoding |
|---|---|---|
| index position | RoPE (paper 0.5081) | PoPE (paper 0.4889) |
| path integration | MapWM | MapPoPE |

5 seeds per arm, one batch, all four arms trained together. Lower NLL is better throughout.

## Registered verdicts

- **R1 (replication)** PoPE - RoPE < 0 and within 0.02 of the paper's -0.0192 gain, and both arms
  within 0.05 NLL of the published values. Failing this, the deviations above are the first suspects.
- **R2 (path integration on the RoPE row)** MapWM - RoPE, paired by seed, MDE = 2.8 sd / sqrt(5).
  **No direction registered.** Music has no action stream and no obvious "move by k" structure, so
  unlike Dyck-2 and Indirect Indexing there is no mechanism argument for a gain; this is the first
  test in this project of path integration on a natural-sequence task with real content.
- **R3 (path integration on the PoPE row)** MapPoPE - PoPE, same test.
- **R4 (interaction)** (MapPoPE - PoPE) - (MapWM - RoPE) with its MDE.
- **Rule 9 check**: r(final training loss, test NLL) across all 20 runs. On a small dataset (229
  pieces, ~52 epochs at this budget) overfitting is the expected failure, so the train/valid/test
  curves are recorded at every evaluation and the gap is reported.
- **Convergence**: the validation curve is recorded throughout; if the best validation step is the
  last step for most runs, the budget is the binding constraint and that is reported, not hidden.

At n=5 the MDE is 1.25 sd. The paper's own effect is 0.019 NLL, so if seed sd exceeds ~0.015 this
design cannot resolve an effect of that size, and the result will be reported as unmeasured.
