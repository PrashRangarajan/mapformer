# Is the sign cost present at each arm's best learning rate? -- results (registered, `LRSWEEP_PREREG.md` + Amendment 1)

Batch 2026-10-16 23:03 -> 10-17 05:03 on GPU 1 only (<= 4 runs, beside another user's job), code 74b24f0, 64 runs, seeds
1500-1507 (fresh). Ring 32 cells, moves +1/-1, observation-step-free SignedAO_r4 (SO) vs AbsAO_r4 (AO, |.| step),
`train_gauge.py --deltas=1,-1`, T = 128, 300 epochs, cosine, wd 0.05; learning rates 1e-3, 3e-3, 1e-2, 3e-2 (grid fixed by
the pilot on seeds 1590-1591: 3e-2 and 1e-1 break both arms, no extension). Independent check before launch
(`docs/audits/2026-10-16/lrsweep_audit.md`, Amendment 1). Analysis `analyze_lrsweep.py` -> `LRSWEEP_ANALYSIS.txt`,
`LRSWEEP_VERDICTS.json`. `experiment_audit`: windowed classes agree (AO 4 SOLVED of 32, SO 21).

**Reading (registered): COST AT BEST RATES** -- each arm at its own best learning rate, the |.| step solves 2/8 against
8/8 (decision p 0.0070 on SOLVED, 0.0057 on accuracy; selection-aware null redone over all 12,870 relabellings). Qualifier
(registered, Amendment 1 MINOR 6): 3 of the |.| runs at its best rate were still DESCENDING at 300 epochs -- budget-scoped.

## B1 (registered primary): best vs best
| arm | best rate (most SOLVED, then lowest final-5% loss) | SOLVED | accuracy T = 128 |
|---|---|---|---|
| SO (signed) | 1e-3 (lowest rate with its best count; 3e-3 also 8/8) | 8/8 | 1.000 |
| AO (|.|) | 3e-3 | 2/8 | 0.963 |

d = AO - SO -0.037; selection-aware p 0.0061 (SOLVED) / 0.0057 (accuracy); naive at the selected rates Fisher 0.0070, perm
0.0014; decision p = max: 0.0070 / 0.0057. Label SIGNED LEARNS BETTER.
- The |.| optimum is bracketed (3e-3 is interior: 1/8 at 1e-3, 1/8 at 1e-2, 0/8 at 3e-2), so "best" is not limited by the grid.
- Stage 2 (decay-matched control, Amendment 1 MAJOR 2) is NOT triggered: AO's best rate is 3e-3 < 1e-2, where AdamW's
  coupled decay is 3x the default's; ROBUST's weight-decay-0 cell (|.| 0/8 vs 8/8) stands as the decay control.
- Threshold sensitivity (loss 0.1, selection redone): 2/8 vs 8/8, same label. Accuracy at 4T at the selected rates: 0.891 vs
  0.998 (selection-aware p 0.0061).
- Cross-fit (declared; rates chosen on half the seeds, tested on the other half): both directions choose the same rates
  (AO 3e-3, SO 1e-3) and give 1/4 vs 4/4 (Fisher p 0.14, n = 4 per arm; perm p 0.029 / 0.14) -- the selection is stable,
  the halves alone are underpowered, as registered.

## Per rate (declared secondary; Holm over the four rates)
| lr | SO SOLVED, acc | AO SOLVED, acc | label (Holm) |
|---|---|---|---|
| 1e-3 | 8/8, 1.000 | 1/8, 0.917 | SIGNED LEARNS BETTER (fresh-seed replication of ROBUST def) |
| 3e-3 | 8/8, 1.000 | 2/8, 0.963 | SIGNED LEARNS BETTER (ROBUST lrx3 had 4/8 vs 8/8, unmeasured) |
| 1e-2 | 4/8, 0.972 | 1/8, 0.872 | UNMEASURED |
| 3e-2 | 1/8, 0.559 | 0/8, 0.584 | UNREADABLE (both arms at or near the 0.50 blank floor) |

SOLVED-vs-lr: signed 8, 8, 4, 1; |.| 1, 2, 1, 0. Higher rates hurt the signed step first and do not help the |.| step past
3e-3. Wrap-cancel share (unweighted / usage-weighted): signed 0.89-1.00 / 0.99-1.00 wherever it learns; |.| 0.05-0.10 /
0.02-0.46 (the 0.46 at 1e-2 is one rate's mean over mostly unsolved runs; no |.| cell exceeds 0.13 unweighted).
Accuracy at 4T: signed 0.998 / 0.976 / 0.930 / 0.561; |.| 0.824 / 0.891 / 0.768 / 0.555. Floors: blank ~0.50, n-gram 0.554,
retrace 0.860 (post hoc, `docs/audits/2026-10-16/ring_floors_out.txt`).

Per seed (accuracy at T = 128, class):
```
seed | 1e-3:SO | 1e-3:AO | 3e-3:SO | 3e-3:AO | 1e-2:SO | 1e-2:AO | 3e-2:SO | 3e-2:AO
1500 | 1.0000 SOLV | 0.9362 STAL | 1.0000 SOLV | 0.9111 STAL | 0.9460 DESC | 0.9936 DESC | 0.4960 STAL | 0.4958 STAL
1501 | 1.0000 SOLV | 0.8759 STAL | 1.0000 SOLV | 0.9662 DESC | 0.9460 STAL | 0.4960 STAL | 0.4960 STAL | 0.4960 STAL
1502 | 1.0000 SOLV | 0.8995 STAL | 1.0000 SOLV | 0.9428 STAL | 0.9407 STAL | 0.9703 DESC | 0.4960 STAL | 0.4960 STAL
1503 | 1.0000 SOLV | 0.9116 STAL | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.6000 STAL | 0.4960 STAL | 0.4960 STAL
1504 | 1.0000 SOLV | 0.9415 STAL | 1.0000 SOLV | 0.9686 STAL | 0.9460 STAL | 1.0000 DESC | 0.4960 STAL | 0.8159 DESC
1505 | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.9993 SOLV | 1.0000 SOLV | 0.9950 DESC | 0.9998 SOLV | 0.4961 STAL
1506 | 1.0000 SOLV | 0.8736 STAL | 1.0000 SOLV | 0.9595 DESC | 1.0000 SOLV | 0.9996 SOLV | 0.4960 STAL | 0.4960 STAL
1507 | 1.0000 SOLV | 0.8971 STAL | 1.0000 SOLV | 0.9587 DESC | 1.0000 SOLV | 0.9184 DESC | 0.4959 DESC | 0.8799 DESC
```

## Consequences
- The learning-rate objection (mock review of draft 5, W1) is answered on the ring: at each arm's best of four rates, with
  the |.| optimum bracketed, the cost remains (2/8 vs 8/8). The title stands, scoped to "each arm's best of four learning
  rates" (prereg reading). ROBUST lrx3's 4/8 was the high end of seed variation (2/8 here on fresh seeds).
- Budget: 3/8 |.| runs at 3e-3 still descending at 300 epochs; the claim stays budget-scoped. Ring only: the torus was
  not swept (ROBUST covers the torus at one rate and a larger model).
