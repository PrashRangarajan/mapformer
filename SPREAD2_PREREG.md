# SPREAD2 -- the exposure test, off the ceiling

Pre-registered 2026-09-12, before any arm was trained. Re-run of `SPREAD_PREREG.md`'s exposure
half, which `SPREAD_RESULTS.md` reports as half-vacuous.

## Why a re-run

SPREAD's registered exposure contrast was `m16_e1200 - m4_e300` = **+0.000 with an MDE of
0.000**, because both arms scored exactly 1.000 on all 8 seeds. A cell that cannot come out any
other way is not evidence, and the analysis still printed CONFIRMED. What DID fire was the
fixed-budget gradient: m4 1.000 > m16 0.996 > m64 0.578, `m4 - m64` = +0.422 (8/8).

So the open question is unchanged: **is the limit queries per token, or the number of query
tokens?** This asks it where accuracy can move in both directions.

## Calibration (already run, `runs/spread_cal/`)

`VanillaEM_P0_r4` on the 4-offset set {1, 4, 16, 64}, 2 seeds per point, primary readout:

| epochs | primary (k in {4,16,64}) |
|---|---|
| 20 | 0.605 |
| 40 | 0.697 |
| 80 | 0.984 |

The ceiling is crossed between 40 and 80. **60 epochs is targeted to land m4 near 0.8**; that is
an interpolation from n=2 points and is stated as such, not measured.

## Arms (all `VanillaEM_P0_r4`, 8 seeds each, ONE batch, `runs/spread2/`)

| arm | offsets | epochs | queries per token |
|---|---|---|---|
| `m4_e60` | {1,4,16,64} | 60 | ~80.6k |
| `m16_e240` | the 16-value set | 240 | ~80.6k -- **exposure-matched to m4_e60** |
| `m16_e60` | the 16-value set | 60 | ~20.2k |
| `m64_e60` | 1..64 | 60 | ~5.0k |

16-value set: 1, 2, 3, 4, 6, 8, 11, 16, 22, 26, 32, 38, 45, 52, 58, 64. Recipe otherwise as
every recency batch: `k_max 64, min_gap 64, T 1024, 48 x 16 per epoch, cosine, lr 1e-3, 1 layer,
d 128, h 2`. Gates for both k-sets already pass at 800 episodes
(`RECENCY_GATES_K4SET.md`, `RECENCY_GATES_K16SET.md`).

Primary readout: mean accuracy over k in {4, 16, 64} at T=1024, paired by seed, MDE = 2.8 sd /
sqrt(8). k=1 excluded because the most-recent shortcut floor scales as 1/m.

## The design check that SPREAD failed -- read FIRST

**`m4_e60` must land in 0.60-0.95.** If it is >= 0.98 the arms are on the ceiling again and the
exposure contrast is NOT interpreted, exactly as in SPREAD; the batch then becomes another
budget-curve point. Any contrast involving an arm at >= 0.98 is labelled ceiling-limited.

## Predictions

- **S5-P1 (exposure is the currency).** `m16_e240 - m4_e60` is BELOW its MDE, with both arms off
  the ceiling. CONFIRMED only if the design check passes -- an agreement between two ceilinged
  arms does not count.
- **S5-P2 (token count costs beyond exposure).** `m16_e240 - m4_e60` is DETECTABLE and NEGATIVE:
  at matched queries per token, serving 16 offsets is still worse than serving 4. P1 and P2 are
  mutually exclusive; if the contrast lands between, both are reported unresolved.
- **S5-P3 (the fixed-budget gradient replicates at a lower budget).** `m4_e60` > `m16_e60` >
  `m64_e60`, with `m4_e60 - m64_e60` DETECTABLE. SPREAD found +0.422 at 300 epochs; this asks
  whether it survives where m4 is not saturated.

## Analysis discipline

`stats_guard.paired` with an MDE beside every contrast ("unmeasured", never "null"), the arm's
position relative to ceiling reported beside every verdict, and `stats_guard.rule9` before any
loss-matched reading -- r(loss, acc) ran **-0.945** on the SPREAD arms, so these accuracy
contrasts are largely fit contrasts and that will be stated again.

## Note on the launcher

SPREAD's wait loops counted every `mapformer.train_recency` process, so the calibration driver
sat idle until an unrelated batch finished. This launcher counts only its own runs (matching the
output directory in argv).
