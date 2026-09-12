# PAIRSPLIT -- resolve PAIRORIGIN's accuracy split

Pre-registered 2026-09-12, before any new seed was trained. Follows `PAIRCONST_RESULTS.md`.

## Why

PAIRORIGIN's +0.280 (EMPair - P0, detectable at n=8) splits into
**+0.182 pathway + 0.098 freedom**, and at n=8 NEITHER half is detectable
(MDEs 0.188 and 0.209). The mechanism attribution is already clean -- the constant-origin control
keeps the per-token rewind route (0.948) while EMPair abandons it (0.189) -- but the accuracy
attribution is unresolved, and the summary files now say so.

## Power, stated before running

Observed sd of the paired difference `EMPair - EMPairConst` is 0.211. MDE = 2.8 sd / sqrt(n):

| n | MDE | vs the +0.098 point estimate |
|---|---|---|
| 8 | 0.209 | hopeless |
| 36 | 0.098 | exactly on the effect -- a null would be uninformative |
| **48** | **0.085** | margin of 0.013 |

So **n = 48**, not the n≈36 I first estimated. Seeds 8-47 for both arms, 80 new runs.

## Arms

`EMPair_r4` and `EMPairConst_r4`, seeds 8-47, one batch, into `runs/pairsplit/`. Recipe as every
recency batch (`k_max 64, min_gap 64, T 1024, 300 ep x 48 x 16, cosine, lr 1e-3, 1 layer, d 128,
h 2`). Seeds 0-7 are reused from `runs/pairorigin` and `runs/pairconst`, licensed by an in-batch
bitwise determinism re-check of `EMPair_r4` s0 (it passed in PAIRCONST, and must pass again).

`VanillaEM_P0_r4` seeds 0-23 already exist in `runs/dof/recency` at this exact recipe, so C2 is
reported at n=24 without new runs.

## Predictions

- **S1 (freedom is real but small).** `EMPair - EMPairConst` at n=48 is DETECTABLE and lands in
  0.05-0.15, i.e. the n=8 point estimate survives at reduced size.
- **S2 (freedom is nothing).** It is below MDE 0.085 -- then per-pair freedom buys no ACCURACY
  at all, and PAIRORIGIN's gain is the pathway. The mechanism finding (route 0.948 vs 0.189)
  would then stand as a change in HOW the model solves it that does not change how well.
- **S3 (the n=8 estimate was low).** It exceeds +0.15.
- **C2 at n=24**: `EMPairConst - P0` predicted DETECTABLE (point estimate +0.182, MDE ~0.109).

**Registered guard against my own pattern**: the first eight seeds in this line have
over-estimated three times (`sep - P0` 1.9x, MagOnly 1.9x, D5). The fresh seeds 8-47 will be
reported ALONE beside the pooled figure (`stats_guard.replication_split`), and if they disagree
with the pooled estimate the fresh ones are what I quote.

## Analysis discipline

`stats_guard.paired` + `replication_split`; `stats_guard.rule9` first -- PAIRORIGIN gave
r = -0.983, so these are fit contrasts and the loss-matched residual cannot separate
optimisation from representation (mediator, `AUDIT_2026-09-10.md` #7).
