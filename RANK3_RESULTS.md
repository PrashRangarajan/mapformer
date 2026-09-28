# Rank 3 per head -- results (2026-09-28)

Pre-registration `RANK3_PREREG.md`; runs `runs/rank3` (+ `runs/rank3_repro`); full output
`RANK3_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank3`); eval `RANK3.md` / `.json`, strata
`RANK3_STRATA.json`, geometry `RANK3_GEOMETRY.md`. Torus, trained AND tested at T=1024, 900 epochs,
8 seeds. E (per-head r=3) is built from our r=2's base at each seed, like B and D.

## Registered verdict: RANK 3 SUFFICES -- on accuracy, not on solved count

| per-head rank | arm | SOLVED (final-5% loss < 0.05) | T=512 | **T=1024** | T=2048 |
|---|---|---|---|---|---|
| 2 | B `Vanilla_r2ph` (stored, `runs/rank_mi`) | 2/8 | 0.910 | 0.885 | 0.832 |
| **3** | **E `Vanilla_r3ph`** (new) | **6/8** | 0.997 | **0.987** | 0.952 |
| 4 | D `Vanilla_r4ph` (stored, `runs/rank_sep`) | 8/8 | 1.000 | 0.999 | 0.970 |

| contrast | SOLVED, Fisher p | T=1024 acc, exact permutation p | Holm (2) | registered |
|---|---|---|---|---|
| rank 3 - rank 2 (E - B) | 6/8 vs 2/8, 0.13 | **+0.102, 0.027** | 0.054 | FIRES |
| rank 4 - rank 3 (D - E) | 8/8 vs 6/8, 0.47 | +0.012, 0.19 | 0.19 | UNMEASURED |

The registered branch (E - B fires on either co-primary, D - E does not, E >= 6/8) is met, at its
boundary on every count: it fires on accuracy only (solved count 6/8 vs 2/8 is p 0.13, as the
pre-registration warned it would be), the Holm-adjusted p over the two contrasts is 0.054, and E
sits exactly at the 6/8 threshold. **Read it as: rank 3 behaves much more like rank 4 than like rank
2, with the rank 2 -> 3 step carrying most of the effect (+0.102 of the +0.114 from 2 to 4).** D - E
UNMEASURED means rank 3 and 4 were not distinguished at n=8, not that they are equal.

Reproduction: D seed 0 retrained in this batch matches the stored per-epoch losses exactly (max
diff 0.0 over 900 epochs), so the comparison with the stored B and D arms is valid.

## The two unsolved seeds sit in the same non-cancelling code as rank 2's stalls
Delta-space geometry per seed (`RANK3_GEOMETRY.md`; opposition |N+S|/scale, 0 = N and S cancel):
the six SOLVED seeds 0.017-0.076; the two unsolved, s0 (final loss 0.243) and s3 (0.134), **1.83 and
1.58**, with |cos(N,E)| 0.99 and 0.65 and one head carrying 76-85% of the action norm. That is the
pattern `RANK_MATCHED_RESULTS.md` found for from-scratch r=2 stalls (opposition 0.87-1.78 on 7/8):
rank 3 lowers how often training falls into the non-cancelling basin, it does not remove the basin.

## What it says about the story
`WHERE_THINGS_STAND.md` framed the fork as "rank 2 is special" (3 solves like 4) vs "rank must reach
4". The result favours the first: the failure is concentrated at a per-head rank equal to the
torus's two degrees of freedom, and one spare direction per head recovers most of it. Scope: torus
(2 DOF), T=1024, n_heads 2, d 128, one recipe, 900 epochs, n=8. Not tested: whether the threshold
tracks the task's DOF (a 3D torus would predict rank 3 fails where rank 4 succeeds; see the ND
environment, `environment_nd.py`).
