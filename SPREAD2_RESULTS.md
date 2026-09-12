# SPREAD2 -- results (pre-registration `SPREAD2_PREREG.md`, commit e75f18f)

4 arms x 8 seeds, one batch, `VanillaEM_P0_r4` on recency with `k` drawn from a set of size m.
Primary readout: mean accuracy over k in {4, 16, 64} at T=1024. The re-run of SPREAD's exposure
half, whose arms were both pinned at exactly 1.000.

## Design check: PASSES

`m4_e60` = **0.913**, inside the registered 0.60-0.95 band and off the ceiling. SPREAD's version
of this cell was 1.000 on 8/8, which is why its exposure contrast could not fire.

## The result

| arm | queries per token | primary | whole trained set | final loss |
|---|---|---|---|---|
| `m4_e60` | 80,600 | 0.913 +/- 0.110 | 0.933 | 0.306 |
| `m16_e240` | 80,600 | 0.957 +/- 0.098 | 0.964 | 0.295 |
| `m16_e60` | 20,200 | 0.590 +/- 0.190 | 0.669 | 1.161 |
| `m64_e60` | 5,000 | 0.248 +/- 0.175 | 0.187 | 2.502 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| m16_e240 - m4_e60 (**exposure-matched**) | +0.045 | 0.168 | 0.166 | 5/8 | unmeasured |
| m4_e60 - m16_e60 (fixed budget) | +0.323 | 0.244 | 0.241 | 7/8 | DETECTABLE |
| m16_e60 - m64_e60 (fixed budget) | +0.342 | 0.252 | 0.249 | 7/8 | DETECTABLE |
| m4_e60 - m64_e60 (fixed budget) | **+0.665** | 0.215 | 0.213 | 8/8 | DETECTABLE |

- **S5-P1 (exposure is the currency): CONFIRMED, and this time not vacuously.** Both arms of the
  matched pair are off the ceiling, the contrast is +0.045, and the test had power to detect
  anything >= 0.166. Sixteen offsets at 240 epochs performs like four offsets at 60 -- nominally
  BETTER, though that difference is itself unmeasured.
- **S5-P2 (token count costs beyond exposure): not confirmed.** Mutually exclusive with P1.
- **S5-P3 (the gradient replicates at a lower budget): MET.** 0.913 > 0.590 > 0.248 with
  m4 - m64 = +0.665 (8/8). At a FIXED budget the number of offsets is decisive; the effect is
  larger here (+0.665) than at 300 epochs (+0.422), as it must be when the budget is smaller.

## What this settles

**The currency is queries per token, not the number of query tokens.** Hold exposure fixed and m
stops mattering; hold the budget fixed and m is worth 0.665. Together with SEARCH -- every solved
cell carries a per-token wrapped rewind, failed cells never do, and one shared k is found on 7/8
seeds -- the account is complete on its own terms: **EM must find one wrapped rewind per query
token, and each token only gets the queries that name it.**

That also explains SPREAD's headline (4x budget moved the full task 0.578 -> 0.928) without
needing a separate mechanism: 4x the budget is 4x the exposure per token.

## Caveats

- `m16_e240` sits at 0.957, close to the band's upper edge, so the matched pair is partly
  compressed. The MDE of 0.166 is the honest statement of what this could have detected.
- r(final loss, accuracy) = **-0.936** over the 32 runs: these are fit contrasts, as every
  contrast on this task has been.
- The exposure axis was varied by the BUDGET, so "queries per token" and "optimiser steps per
  token" are not separated here. Separating them needs the same exposure at different batch
  sizes, which nothing in this line has run.
