# Does a start past pi rescue the non-negative step on the torus? -- results (2026-10-15)

Pre-registration `TORUS_START_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-14/torus_start_audit.md`).
Batch 2026-10-15 01:40-07:51 at 15047f7 on both GPUs; torus 64x64, T = 1024, 900 epochs, observation-step-free 4-move models,
seeds 1200-1207, n = 8. Registered output `TORUS_START_ANALYSIS.txt`, `TORUS_START_VERDICTS.json`. Floors: blank 0.507, best
n-gram 0.576.

| arm | step | start | acc T = 1024 | SOLVED | classes | strata: lag < 128 / >= 128 / wrap-only |
|---|---|---|---|---|---|---|
| SO4 | signed | default | 1.000 | 8/8 | all SOLVED | 1.000 / 1.000 / 1.000 |
| AO4 | non-negative | default | 0.896 | 1/8 | 7 STALLED | 0.939 / 0.730 / 0.632 |
| AO4_high | non-negative | large, pair sums > pi | 0.900 | 0/8 | 5 DESCENDING, 3 STALLED | 0.949 / 0.777 / 0.534 |
| SO4_high | signed | the same large angles | 0.919 | 2/8 | 5 DESCENDING, 1 STALLED | 0.970 / 0.716 / 0.627 |
| AO4_mid | non-negative | large, pair sums < pi | 0.912 | 0/8 | 5 DESCENDING, 3 STALLED | 0.950 / 0.809 / 0.621 |

## Registered
- **TS1 (primary): NO EFFECT DETECTED** -- AO4_high 0/8 vs AO4 1/8, +0.005 [95% CI -0.060, +0.070] (p 0.89). **The start past pi
  does not rescue the non-negative step on the torus** (prediction RESCUES not met).
- **TS2: UNMEASURED** (AO4_high vs AO4_mid -0.011, p 0.71). **TS3: UNREADABLE** (SO4_high solves 2/8 < 4). **TS4: SIGNED
  LEARNS BETTER** -- AO4 1/8 vs SO4 8/8 (Fisher p 0.0014), -0.104 [-0.150, -0.060]: the sign cost replicates.

## Secondaries (declared; no verdict)
- **The large start hurts the signed model too** on the torus: SO4_high 2/8 (0.919) vs SO4 8/8 (not a registered contrast).
  From the large start both signs end alike (0.90-0.92, wrap-only revisits 0.53-0.63) and most runs are still DESCENDING at 900
  epochs (5/8 per large-start arm): the large start is slow on the torus, within this budget.
- Wrap share at the end: SO4 1.00; every other arm 0.05-0.19.

## What it means
- **The mechanism's start lever is ring-only (within 900 epochs).** On the ring a start past pi made the non-negative step learn
  the map 8/8 with no sign cost (MECHANISM); on the torus the same start leaves it at 0/8 -- and also drags the signed model
  from 8/8 to 2/8. So on the torus the large random start is itself a bad start, for either sign: with 64 cells per axis and
  two opposite pairs sharing every channel, a random position code of large angles is far from any map, and training is
  still descending at the end. The sign cost at the default start replicates (1/8 vs 8/8).
- What survives on the torus: the sign cost, existence (GENERALITY: the exact non-negative solution is re-found under
  training), and the gauge / cut-circle geometry (Appendix). What does not transfer: "a start past pi fixes it".
- For the paper: state the start manipulation as a ring result; the torus shows the limit of that lever (a large start is
  not a good start on a larger map). The design implication ("initialise near the wrap") is withdrawn as a general claim.

## Caveats
900 epochs with most large-start runs still descending (budget-scoped: a longer run might separate them); one start
distribution; n = 8.
