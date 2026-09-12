# SPREAD -- results (pre-registration `SPREAD_PREREG.md`, commit 4cf44c2)

4 arms x 8 seeds + a bitwise determinism repro, one batch, `VanillaEM_P0_r4` on recency.
Primary readout: mean accuracy over k in {4, 16, 64} at T=1024 (k=1 excluded because the
most-recent shortcut floor scales as 1/m). Determinism re-check: repro s0 **bitwise identical**
to the stored `runs/dof/recency` s0, so reusing the stored arm as `m64_e300` is licensed.

## The result

| arm | queries per token | primary | per seed | whole trained set | final loss |
|---|---|---|---|---|---|
| m4_e300 | 403k | **1.000 +/- 0.000** | 1.000 x8 | 1.000 | 0.048 |
| m16_e1200 | 403k | **1.000 +/- 0.000** | 1.000 x8 | 1.000 | 0.071 |
| m16_e300 | 101k | **0.996 +/- 0.010** | 6 at 1.000, min 0.971 | 0.978 | 0.245 |
| m64_e1200 | 101k | **0.928 +/- 0.131** | 6 at ~1.000, **two at 0.724 / 0.707** | 0.930 | 0.359 |
| m64_e300 (stored) | 25k | **0.578 +/- 0.123** | 0.407 .. 0.736 | 0.609 | 1.340 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| m16_e1200 - m4_e300 (exposure-matched) | +0.000 | 0.000 | 0.000 | 0/8 | **ceiling, uninformative** |
| m64_e1200 - m16_e300 (exposure-matched) | -0.068 | 0.133 | 0.132 | 1/8 | unmeasured |
| m4_e300 - m16_e300 (fixed budget) | +0.004 | 0.010 | 0.010 | 2/8 | unmeasured |
| m16_e300 - m64_e300 (fixed budget) | **+0.418** | 0.125 | 0.124 | 8/8 | DETECTABLE |
| m4_e300 - m64_e300 (fixed budget) | **+0.422** | 0.123 | 0.122 | 8/8 | DETECTABLE |

## Registered verdicts, reported as they came out

- **S4-P2 (spread gradient at fixed budget): MET.** m4 1.000 > m16 0.996 > m64 0.578, and
  m4 - m64 = +0.422 (8/8, DETECTABLE). **At a matched budget, the fewer distinct offsets the
  model must serve, the better it does.** This is the registered test and it fired.
- **S4-P1 (exposure): the analysis prints CONFIRMED; read it as HALF-VACUOUS.** The first
  matched pair is +0.000 with an MDE of 0.000 because `m4_e300` and `m16_e1200` are BOTH exactly
  1.000 on all 8 seeds -- a ceiling-against-ceiling comparison that could not have come out any
  other way, which is the pre-registration error this project already recorded once (a verdict
  cell that cannot fire). Only the second pair is informative: `m64_e1200 - m16_e300` = -0.068
  (MDE 0.132), i.e. consistent with exposure but **unmeasured**, and carried entirely by the two
  failing seeds. The honest statement is: **matching queries-per-token is CONSISTENT with the
  arms landing together, and the design could not distinguish that from a ceiling.**
- **S4-P3 (mechanism): NOT MET.** r(rewind fraction, primary) = +0.573 over 40 cells against
  >= 0.70 registered. The clearest discrepancy is `m4_e300`: accuracy exactly 1.000 on every
  seed with rewind fractions of 0.25-0.75. With only 4 trained query tokens the fraction is
  quantised to quarters, but at ceiling accuracy half the tokens show no rewind by this readout,
  so either small-m models solve some offsets another way or the readout misses a route.

## The finding that is not in any registered verdict

**Budget alone moves the full task from 0.578 to 0.928.** The standard 64-offset condition at
4x the epochs reaches ~1.000 on 6 of 8 seeds (the other two stick at 0.71). So EM's recency
deficit is to a large extent a STEP-EFFICIENCY gap, not a capability gap -- which is what a
per-token search account predicts, since every token needs its own wrapped rewind and gets
1/64 of the queries.

**Two limits on that, stated rather than buried.**
1. **WM at 1200 epochs was never run.** "EM needs ~4x the budget to approach WM's 300-epoch
   0.975" describes EM's own curve, not a matched comparison. The EM-WM gap of -0.375 is at a
   SHARED 300-epoch budget, which is the fair comparison; but it now looks substantially like a
   convergence-rate difference, and a matched-budget-at-1200 arm would say how much survives.
2. **Rule 9 bites hard here**: r(final loss, accuracy) = **-0.945** over the 40 runs
   (acc = 1.040 - 0.338*loss, resid sd 0.059). These accuracy differences are largely fit
   differences. That is consistent with the search/optimisation account rather than a
   representational one -- but it means the accuracy contrasts carry little information the
   training loss does not already carry.

## What this does and does not settle

- **Settles:** at a fixed budget, serving more distinct offsets is worse, monotonically, and the
  effect is large (+0.422). The obstacle is per-token, as SEARCH concluded.
- **Does not settle:** whether "queries per token" is the right currency, because the design put
  three of five arms on the ceiling. To separate exposure from token count the arms must sit
  where accuracy can move in both directions -- e.g. m4 / m16 / m64 at a budget chosen so the
  m4 arm lands near 0.8, not 1.0. That is the re-run this result asks for.
- **Also open:** why two `m64_e1200` seeds stall at 0.71 with high final loss while six reach
  ~1.000. Bimodal basin selection, the same shape as `H12_BUDGET_CURVE.md`'s nb=4000 case.

## Process note

The analysis first crashed on its own guard: it compared each checkpoint's stored `k_set`
against the arm definition and reported a mismatch on a CORRECT batch, because
`ckpt_guard._FLAT_CONFIG_KEYS` is an allow-list that never learned the `k_set` key the trainer
now saves. The batch was fine; the reader was not. The allow-list is widened and the guard
tests still pass 13/13.
