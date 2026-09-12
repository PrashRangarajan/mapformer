# PAIRCONST -- is PAIRORIGIN's +0.280 per-pair freedom, or 2,048 parameters?

Pre-registered 2026-09-12, before any arm was trained. Control for `PAIRORIGIN_RESULTS.md`.

## The confound

`EMPair_r4` beats `VanillaEM_P0_r4` by **+0.280** (7/8, MDE 0.216) on varying-k recency, and its
phase spread across pairs goes 0.000 -> 1.448 while its per-token-rewind route drops 0.962 ->
0.185. But it also adds **2,048 parameters** in the position pathway. This line has been caught
by exactly that substitution before: `AlignLock` vs `AlignFree` differed in the optimiser
treatment of a parameter as well as in the freedom granted, and `MagOnly` had to be built to
separate them (`AUDIT_2026-09-10.md` #8, rule 31).

## The control

`EMPairConst_r4` (`model_em_pairconst.py`): the same pathway, reading a LEARNED CONSTANT instead
of the token embedding.

    EMPair       q^p_t = p0 + W_out W_in x_t   -> origins vary per token   (per-pair kernel)
    EMPairConst  q^p_t = p0 + W_out W_in c     -> origins identical for every token (ONE kernel)

Verified at construction: identical function to `VanillaEM_P0_r4` at init (max |logit diff|
**0.000e+00**), pathway alive and trained on the same scale, and **224,537 parameters against
EMPair's 224,409 -- the control has 128 MORE**, so a null here excludes capacity a fortiori.
Its origins are constant across tokens by construction at every step, not merely at init.

## Arms

`EMPairConst_r4` x 8 seeds, plus `EMPair_r4` seed 0 retrained as a **bitwise determinism
re-check** against `runs/pairorigin/EMPair_r4_s0` (`train_variant.py` gained a registry line
since that batch; reuse of the stored `EMPair_r4` and `VanillaEM_P0_r4` arms as comparators is
licensed only if it is identical). Recipe as every recency batch: `k_max 64, min_gap 64, T 1024,
300 ep x 48 x 16, cosine, lr 1e-3, 1 layer, d 128, h 2`. Output `runs/pairconst/`.

Primary readout: held-out accuracy at T=1024, paired by seed, MDE = 2.8 sd / sqrt(8).

## Manipulation checks, before any verdict

1. Determinism re-check bitwise identical, else the comparators are retrained and nothing is
   read from the stored batch.
2. Origins constant across tokens in the trained `EMPairConst` checkpoints (spread exactly 0)
   and NON-constant in `EMPair` -- the manipulation is the content dependence, so it must be
   shown to be absent here and present there.
3. The origin pathway moved (`assert_moved` on `q_origin_out` / `k_origin_out`).

## Predictions

- **C1 (freedom, not parameters).** `EMPair - EMPairConst` >= **+0.20** and DETECTABLE.
- **C2.** `EMPairConst - P0` is BELOW its MDE: the parameters alone buy nothing.
- **C3 (the alternative).** `EMPairConst - P0` >= +0.20 and DETECTABLE -> the gain was capacity
  or optimiser slack, **PAIRORIGIN's kernel-sharing reading is WITHDRAWN**, and P1 goes back to
  unresolved. C1 and C3 are mutually exclusive.
- Anything between: reported as unresolved, with both contrasts and their MDEs.

## Analysis discipline

`stats_guard.paired` with an MDE beside every contrast; `stats_guard.rule9` first -- PAIRORIGIN
gave r(loss, acc) = -0.983, so these are fit contrasts and the loss-matched residual cannot
separate optimisation from representation (mediator, `AUDIT_2026-09-10.md` #7).
