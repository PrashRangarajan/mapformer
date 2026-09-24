# Warm-start stability test -- results (2026-09-24)

Pre-registration `RANK_PROJ_PREREG.md`. Runs `runs/rank_proj` (frozen), `runs/rank_proj_train`
(trainable). Full output `RANK_PROJ_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank_proj`).

## Registered verdict: S1 STABLE

| arm | SOLVED | STALLED | DESCENDING | T=1024 accuracy |
|---|---|---|---|---|
| **TRAINABLE**: r=2 started from the rank-2 projection of a solved r=4 | **7** | 0 | 1 | 0.994 |
| **CONTROL**: r=4 re-trained from its own solved weights, same recipe / seed / data | 7 | 0 | 1 | 0.996 |
| reference: r=2 continued from its own unsolved 900-epoch weights | 1 | 3 | 4 | 0.905 |

Primary: SOLVED 7/8 vs 7/8, Fisher p 1.00. Accuracy trainable vs control: permutation p 0.67;
trainable vs the r=2 reference: p 0.0009.

**FROZEN** (existence): the projected r=2 models score 0.9955 at T=1024 against their r=4
sources' 0.9966, >= 0.95 on 8/8 seeds (min 0.981). At T=2048 the projection loses more:
0.980 -> 0.957 (seed 4: 0.986 -> 0.880).

## What it says

- **r=2 holds the solution under training.** The restart to lr 1e-3 raised every trainable
  run's TRAINING loss to 0.37-0.46 (control 0.26-0.30), and 7/8 came back below 0.05 at epochs
  553-654, the window of the control (497-669). A review re-ran seeds 0 and 6 with a snapshot
  every 25 epochs (bit-exact with the committed runs): during the kick T=1024 accuracy fell to
  0.885-0.95 and the long-range strata to near the floor, but **the action code stayed
  cancelling throughout** (opposition 0.008-0.12, |cos| <= 0.18), where from-scratch r=2 stalls
  sit at opposition 0.79-1.78. Held-out NLL at the training-loss peak was 0.045 against a
  training loss of ~0.22. So the kick is high-LR noise around the solution, not an exit from
  it: r=2 **keeps a cancelling code through a perturbation that costs long-range accuracy, and
  re-anneals**. (CORRECTED after review: an earlier version said it "recovers the solution
  after being knocked out", which overstates.)
- **So r=2's from-scratch failure is SEARCH** (rule 32: exists, stable, not found). The solution
  exists (frozen), is held under training once reached (trainable), and r=2 trained from
  scratch does not reach it: 1/8 after 900 + 900 epochs. What this does NOT show is that r=2
  could leave the non-cancelling configurations its from-scratch runs stall in; the test never
  goes there.
- Exactly the pattern of EM's recency deficit (`SEARCH_RESULTS.md`): present, stable,
  not found. The extra latent dimensions of r=4 change what training finds, not what the
  model can represent.

Caveats: one task, one width (d=128), one recipe (batch 16, lr 1e-3, warm restart); our
bottleneck is shared across heads where the paper's is per head, so "r=2" here has half the
paper's latent dimensions (our r=2 inside the paper's r=2 inside our r=4). The kick is larger
at r=2 than at r=4, so "as stable as r=4" is about the endpoint, not the path. `w_out`'s
default init scales as 1/sqrt(r), so dimension count and parameterisation scale are not
separated (rule 15).
