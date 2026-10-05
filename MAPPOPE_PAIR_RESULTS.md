# MapPoPE's two changes, separated -- results (2026-10-05)

Pre-registration `MAPPOPE_PAIR_PREREG.md` (+ Amendment 1 from the code audit, committed 49fb251 before launch). Runs
`runs/mappope_pair/p0` (72 runs, one batch: rank-2 arms seeds 10-25, rank-4 arms seeds 10-17, all fresh); registered
output `MAPPOPE_PAIR_ANALYSIS.txt` (`analyze_mappope_pair.py`, run by the driver after a second md5 check); evals
`MAPPOPE_PAIR_R2.json` / `_R4.json`. Pilot: the wrapper reproduces paper2x2's stored Vanilla and MapPoPE-Flat s0 runs
bit for bit (300/300 epochs).

## Registered: SCORE RULE -- PoPE's score carries the whole gain; the doubled angle count adds nothing

Paper torus, T=128 (matched length), 1 layer, held-out map, rank 2, n = 16 per arm:

| arm | angles per head | T=128 accuracy | SOLVED | revisit NLL | [T=512, T=1024: no verdict] |
|---|---|---|---|---|---|
| MapWM r2 | 32 | 0.9752 +/- 0.0393 | 10/16 | 0.0564 | [0.936, 0.831] |
| MapPoPE r2, pairwise (new) | 32 | **0.9995** +/- 0.0010 | **16/16** | 0.0014 | [0.977, 0.930] |
| MapPoPE r2 (current) | 64 | 0.9997 +/- 0.0008 | 16/16 | 0.0008 | [0.980, 0.933] |

Floors: best n-gram 0.598, always-blank 0.507.

| contrast | d | perm p | 95% CI | verdict |
|---|---|---|---|---|
| TOTAL = MapPoPE 64 - MapWM | +0.0244 | 0.0090 | [+0.0047, +0.0450] | fires (replicates paper2x2's +0.028 on fresh seeds) |
| SCORE = MapPoPE 32 - MapWM | +0.0243 | 0.0120 | [+0.0045, +0.0449] | fires |
| COUNT = MapPoPE 64 - MapPoPE 32 | +0.0002 | 0.60 | **[-0.0004, +0.0008]** | does not fire; bounded near zero |

CIs: `docs/audits/2026-10-05/mappope_pair_ci.py` / `_out.txt` (descriptive, computed after the verdict). SOLVED 16/16
vs 10/16 (Fisher p 0.018) for both TOTAL and SCORE. NLL tells the same story (SCORE -0.0550, p 0.012; COUNT -0.0006,
p 0.47). The angle count's contribution is not just "unmeasured": its CI excludes anything above 0.0008, about 3% of
the total effect. r(final loss, acc) over 72 runs -0.963.

## Secondaries (no verdict)
- **Rank 4: every arm is at 1.000 on every seed** (CEILING; nothing to decompose).
- **The frequency count does NOT explain MapPoPE's small rank-4 gain.** The rank-4 upgrade at T=1024 (OOD, rule 10:
  robustness, not capability): MapWM +0.092 (p 0.026), pairwise MapPoPE +0.020 (p 0.14), MapPoPE 64 +0.027 (p 0.041).
  The pairwise arm behaves like MapPoPE, not like MapWM: the small gain comes with PoPE's score, which already makes
  rank 2 work, not with the 64 angles (the suspect named in `MAPPOPE_R4_RESULTS.md`, now cleared).
- Out of distribution the score rule's lead grows (T=512 +0.041, p 0.0025; T=1024 +0.099, p 0.0013); the angle count
  stays at zero (+0.003, +0.003).
- Dropout-scale check (DROPOUT_RESCORE): paper-torus path arms move <= 0.0007 under the correction, so eval mode does
  not manufacture these contrasts.

## What it means
- MapPoPE's advantage over MapWM on the paper torus is the score rule -- non-negative content magnitudes, phase set by
  position alone, so content cannot move the attention peak -- not the extra frequencies. Every earlier MapPoPE vs
  MapWM comparison that changed both can now be read as a score-rule comparison on this task.
- **PoPE's score rescues rank 2**: 16/16 SOLVED against 10/16 for MapWM at the same rank and angle count. The rank
  theory (`docs/theory/2026-10-04/00_PLAN.md` T1) says a rank-D head fails by drifting or by collapsing an axis; a score
  in which content cannot shift the phase peak may tolerate one of those failure modes. This is a hypothesis, untested:
  the direct test is PoPE's score at rank 2 on the T=1024 torus, where MapWM r2 solves 0/8 and r4 8/8
  (`RANK_SEP_RESULTS.md`).

## Caveats
- One task (paper torus), one length (T=128), 1 layer, 2 heads, n = 16 (rank 2) / 8 (rank 4). The total effect is
  small (+0.024) because MapWM r2 is already near ceiling on 10 of 16 seeds; the result is about which change carries
  it, not about its size.
- SCORE is a bundle (non-negative magnitudes, no content phase except the limited one from untied per-element deltas,
  64 learned deltas, positive score mean); which part of the bundle matters is not separated.
- The rank-4 OOD readout mixes seeds 10-25 (r2) with 10-17 (r4); it is a secondary.
