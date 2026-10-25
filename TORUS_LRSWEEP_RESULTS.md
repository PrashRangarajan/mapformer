# Is the sign cost present on the T = 1024 torus with each step at its own best learning rate? -- results (G-4, registered, `TORUS_LRSWEEP_PREREG.md` + Amendment 1)

Batch 2026-10-24 06:49 -> 21:14 on **kalman** GPU 1 (two runs at a time), code `fb7a028` (md5 guard over 102 files,
`runs/torus_lrsweep/code_md5.txt`, re-checked before every launch), torch 2.10.0+cu128, 48 launches for 48 runs (no relaunch,
no failure in `torus_lrsweep.log`), marker `.torus_lrsweep_done`. Fresh seeds 2400-2407. **Task and recipe:** GENERALITY's
torus recipe unchanged except the rate: 64 x 64 torus, 16 object types, p_empty 0.5, T = 1024 moves, batch 16, 98 batches per
epoch, 900 epochs, cosine (5% warm-up, floor 0.1 x lr), AdamW wd 0.05, 1 layer, 2 heads, d 128, `--fast-attn`. **Arms:**
observation-step-free SignedAO4_r4 (SO, signed step) and AbsAO4_r4 (AO, |.| step; only the four move tokens step; shared
rank-4 latent), at lr 1e-3, 3e-3, 1e-2. At a seed every arm and rate shares the initial weights and the training stream.
Independent results-blind check before launch (`docs/audits/2026-10-24/g4_torus_lrsweep_audit.md`; 2 MAJOR, 9 MINOR, all
folded into Amendment 1). Analysis `analyze_torus_lrsweep.py` -> `TORUS_LRSWEEP_ANALYSIS.txt`, `TORUS_LRSWEEP_VERDICTS.json`;
evaluations `runs/torus_lrsweep/eval/` (held-out map, env seed 10000, 100 walks at T = 1024 and 2048; revisit strata).
**Host:** the prereg's launch line names koopman; the batch ran on kalman (same driver, same commit, GPU 1). The extension
pilot ran on koopman (seeds 2492-2493) and the timing pilot on kalman (seed 2490); neither is pooled.
**Floors** (T = 1024 eval stream, post hoc, `docs/audits/2026-10-20/WHAT_WHERE_T1024.md`): blank 0.506, best n-gram 0.575,
retrace 0.843.

**Reading (registered): COST AT BEST RATES** [5 AO runs still DESCENDING at AO's best rate: budget-scoped]. Each step at its
own best of three learning rates, the |.| step solves **0/8** (best rate 3e-3, accuracy 0.890) against the signed step's
**8/8** (best rate 1e-3, accuracy 1.000): d = -0.110, decision p 0.0002 on SOLVED and 0.0095 on accuracy (selection redone
inside the null over all 12,870 relabellings; decision p = max(selection-aware, naive)). Label SIGNED LEARNS BETTER. Stage 2
(decay control) not triggered: AO's selected rate is below 1e-2. The best-rate result now holds on the ring (LRSWEEP) and on
the T = 1024 torus.

## B1 (registered primary): best vs best

| arm | best rate (most SOLVED, then lowest mean final-5% loss, then lower rate) | SOLVED | accuracy T = 1024 |
|---|---|---|---|
| SO (signed) | 1e-3 (the grid's lowest rate; the only rate where SO solves) | **8/8** | **1.000** |
| AO (\|.\|) | 3e-3 (0/8 at every rate; chosen by the loss tie-break, 0.488 vs 0.513 at 1e-3 and 1.036 at 1e-2) | **0/8** | **0.890** |

d = AO - SO -0.1097. Selection-aware p (exact, 12,870 relabellings of the 16 rate profiles, selection redone in each):
SOLVED 0.0002, accuracy 0.0095. Naive at the selected rates: Fisher 0.0002, permutation 0.0002. Decision p = max: SOLVED
0.0002, accuracy 0.0095 (both routes fire; the accuracy route also needs d <= -0.05, met). UNSTABLE and unsolved at the
selected rates: AO 0/8, SO 0/8. DESCENDING at AO's best rate: 5/8.
- **Edges.** SO's selected rate is the grid's lowest, but SO is at ceiling there (8/8, 1.000 on every seed), so a lower rate
  cannot improve the signed side and the edge does not matter for the claim. AO's selected rate is interior: 3e-3 has AO's
  lowest selection score and its highest accuracy (0.890 vs 0.855 at 1e-3 and 0.768 at 1e-2), so the choice does not rest
  on the tie-break, and AO solves 0/8 at every rate, so no tested rate changes the SOLVED contrast.
- **Grid.** The extension pilot (koopman, seeds 2492-2493, outside the batch; `docs/audits/2026-10-24/g4_pilot_PILOT_ANALYSIS.txt`,
  copied read-only from `runs/torus_lrsweep_pilot/PILOT_ANALYSIS.txt` there) gave AO at 1e-2 0/2 SOLVED (mean selection score
  0.633) and at 3e-2 0/2 with both runs UNSTABLE (2.020, at the memoryless loss) -> DO NOT EXTEND (Amendment 1 MINOR 1 rule).
- **Stage 2** (wd = 5e-5 / lr at AO's selected rate) triggers only when AO's selected rate is >= 1e-2 (Amendment 1 MAJOR 1);
  at 3e-3 the coupled decay exponent is 7.2 (3x the default's 2.4), and no decay control is run (prereg Sec. 3: stated as a
  limit; ROBUST's wd-0 control is ring-only).

## Per rate (declared secondary; ROBUST cell rule on Holm-adjusted p, Holm over the 3 rates per route)

| lr | SO SOLVED, acc | AO SOLVED, acc | d | Fisher p (Holm) | perm p (Holm) | label |
|---|---|---|---|---|---|---|
| 1e-3 | 8/8, 1.000 | 0/8, 0.855 | -0.145 | 0.0002 (0.0005) | 0.0002 (0.0005) | SIGNED LEARNS BETTER (fresh-seed replication of GENERALITY E3) |
| 3e-3 | 0/8, 0.975 | 0/8, 0.890 | -0.085 | 1.0 (1.0) | 0.0002 (0.0005) | UNREADABLE (SO SOLVED 0 < 4) |
| 1e-2 | 0/8, 0.968 | 0/8, 0.768 | -0.200 | 1.0 (1.0) | 0.0002 (0.0005) | UNREADABLE (SO SOLVED 0 < 4) |

SOLVED-vs-lr: signed 8, 0, 0; |.| 0, 0, 0. On the torus a higher rate hurts the signed step first, as on the ring, but more
sharply: at 3e-3 and 1e-2 every signed run is still DESCENDING at 900 epochs (accuracy 0.975 / 0.968; 3/8 below loss 0.1 at
3e-3). Signed is ahead in accuracy at every rate (permutation p 0.0002 each), but the registered rule gates on SO >= 4/8, so
only the 1e-3 cell is readable per rate. The decay exponent sum(lr_t) x wd over the run is 2.4 / 7.2 / 24.1 by rate.

**UNSTABLE (Amendment 1 MAJOR 2; RUNAWAY = an epoch after warm-up above H_map(seed) + 0.05 or a final move step > 1e3;
COLLAPSED = final-5% loss > 1.9).** 2 of 48 runs, both AO at 1e-2, neither at a selected rate: s2404 (max post-warm-up loss
- H_map +0.008, H_map 2.054; final-5% loss 2.022, i.e. COLLAPSED at the memoryless level; step max 0.00; STALLED, accuracy
0.514 at the blank floor) and s2406 (+0.105 over H_map 2.094, RUNAWAY by loss; final-5% 0.870; step max 25.8; STALLED). No
SO run is UNSTABLE and none diverged.

## Declared secondaries (no verdict depends on them)

- **Threshold sensitivity** (SOLVED at loss 0.1, selection redone): AO 3e-3 0/8 vs SO 1e-3 8/8; decision p SOLVED 0.0002,
  accuracy 0.0073 -> SIGNED LEARNS BETTER.
- **Accuracy at T = 2048** (2x the training length; robustness, not capability) at the B1-selected rates: AO 0.817 vs SO
  0.998, d -0.182, selection-aware p 0.010.
- **Cross-fit** (rates selected on half the seeds, tested on the other half; n = 4 per arm, where Fisher reaches p < .05
  only at 4/4 vs 0/4): selected on 2400-2403 (AO 3e-3, SO 1e-3), tested on 2404-2407: 0/4 vs 4/4 (0.844 vs 1.000), Fisher
  and perm p 0.029; selected on 2404-2407 (AO 1e-3, SO 1e-3), tested on 2400-2403: 0/4 vs 4/4 (0.855 vs 1.000), p 0.029.
  AO's selected rate changes between halves (it solves 0 at every rate, so the loss tie-break decides); the contrast does not.
- **Revisit strata at T = 1024** (mean over seeds; plain lag < 128 | plain lag >= 128 | wrap-only):

| lr | SO | AO |
|---|---|---|
| 1e-3 | 1.000 / 1.000 / 1.000 | 0.913 / 0.620 / 0.513 |
| 3e-3 | 1.000 / 0.973 / 0.714 | 0.957 / 0.618 / 0.495 |
| 1e-2 | 1.000 / 0.921 / 0.684 | 0.825 / 0.489 / 0.490 |

  |.| keeps short-gap revisits and sits near the 0.507 floor on wrap-only revisits at every rate; the signed runs left
  unsolved at the higher rates lose wrap-only revisits too (0.68-0.71).
- **Wrap-cancel share** (wrap-aware; unweighted / usage-weighted) and **opposite-sign share**: signed 1.00 / 1.00 and 1.00 at
  every rate; |.| 0.11 / 0.36 and 0.09 (1e-3), 0.23 / 0.66 and 0.19 (3e-3), 0.08 / 0.61 and 0.06 (1e-2).
- **Clock / map ratio** (audit-fixed definition, `clock_table.py`; usage-weighted, symmetric map; run medians): signed
  0.001 / 0.002 / 0.004 by rate (solved at 1e-3, unsolved above); |.| 1.143 / 0.033 / 0.650, all unsolved. Near-pi usage
  share: signed 0.00, |.| 0.09 / 0.09 / 0.03. At 3e-3 the |.| runs have a small clock and a usage-weighted wrap-cancel share
  of 0.66, yet none solves and their long-gap and wrap-only revisits stay at 0.62 / 0.50: descriptive, no account read from it.
- **Loss AUC / mean selection score**: SO 0.193 / 0.002, 0.297 / 0.130, 0.379 / 0.174; AO 0.762 / 0.513, 0.787 / 0.488,
  1.614 / 1.036 (1e-3 / 3e-3 / 1e-2).
- **Run classes**: SO 8 SOLVED at 1e-3, 8 DESCENDING at 3e-3 and at 1e-2; AO 6 STALLED + 2 DESCENDING (1e-3), 3 + 5
  (3e-3), 3 + 5 (1e-2).

## Cross-batch references (labelled, never pooled; rule 12)

- GENERALITY E3 (seeds 700-707, same recipe, lr 1e-3): AO4 0/8, 0.880 vs SO4 8/8, 1.000. This batch's 1e-3 cell on fresh
  seeds: 0/8, 0.855 vs 8/8, 1.000 -- replicated.
- ROBUST torus (seeds 1400-1407, 2 layers, d 256, lr 1e-3): |.| 2/8, 0.967 vs signed 8/8, 0.999.
- LRSWEEP ring (seeds 1500-1507, four rates 1e-3..3e-2): |.| best 3e-3 2/8, 0.963 vs signed best 1e-3 8/8, 1.000.

## Predictions (prereg Sec. 4) against outcomes

| prediction | outcome |
|---|---|
| SO 8/8 at 1e-3 (replicates GENERALITY E3) | 8/8, 1.000: as predicted |
| SO most likely 7-8/8 at 3e-3; 1e-2 uncertain | **missed**: 0/8 at 3e-3 and at 1e-2, every run DESCENDING (0.975, 0.968) -- on the ring signed solved 8/8 at 3e-3 |
| AO 0-1/8 at 1e-3 | 0/8, 0.855 |
| AO's best rate most likely 3e-3, with <= 3/8 | 3e-3, 0/8 |
| 1e-2 unlikely to be AO's best (timing pilot collapsed) | not best; 2/8 UNSTABLE there |
| COST AT BEST RATES most likely (~0.6), UNMEASURED next (~0.3), COST CLOSES least likely (~0.1); stage 2 unlikely | COST AT BEST RATES; stage 2 not triggered |

## Checks on the reading

- **`experiment_audit`** (rule 3), `python3 -m mapformer.experiment_audit --runs-dir mapformer/runs/torus_lrsweep` from
  /home/prashr (read-only; also per rate): exits 2 -- no (json, pt) pairs with key 512 in its layout, so it runs check [1]
  only. Its windowed classes agree with the registered per-seed classes in every cell: pooled AO 12 STALLED + 12 DESCENDING,
  SO 8 SOLVED + 16 DESCENDING; per rate AO 6 + 2 / 3 + 5 / 3 + 5 (STALLED + DESCENDING), SO 8 SOLVED / 8 DESCENDING / 8
  DESCENDING. Its "not all arms converged" flag concerns AO at every rate and SO at 3e-3 and 1e-2; the selected cells are
  SO 1e-3 (all flat, all SOLVED) and AO 3e-3 (5 DESCENDING: the registered budget qualifier).
- **Numbers re-derived** from the checkpoints and evaluation JSONs (scratch script, not committed): every cell's mean
  accuracy at T = 1024 and 2048, SOLVED at 0.05 and 0.1, mean final-5% loss and the stored lr / 900 epochs match the
  analysis to the printed digits.
- **Rule 2:** r(log10 final loss, accuracy) = -0.646 at T = 1024 and -0.763 at T = 2048 over the 48 runs. The primary fires
  on SOLVED (0/8 vs 8/8) as well as on accuracy.
- **Power (prereg Sec. 5):** with SO at 8/8, the SOLVED route fires for AO@best <= 3/8; 0/8 is the strongest reachable outcome.
- **Evaluation mode** under-reports one-layer runs below ceiling (attention-dropout scale, `DROPOUT_RESCORE.md`); no re-score
  was declared; both arms are read the same way, and SOLVED does not depend on it.

## Per seed (accuracy at T = 1024 and class: SOLV SOLVED, STAL STALLED, DESC DESCENDING; ! = UNSTABLE)

```
seed | 1e-3:SO     | 1e-3:AO     | 3e-3:SO     | 3e-3:AO     | 1e-2:SO     | 1e-2:AO
2400 | 1.0000 SOLV | 0.8605 STAL | 0.9782 DESC | 0.9288 DESC | 0.9466 DESC | 0.8288 DESC
2401 | 1.0000 SOLV | 0.8948 STAL | 0.9622 DESC | 0.9546 DESC | 0.9966 DESC | 0.7768 DESC
2402 | 1.0000 SOLV | 0.8233 STAL | 0.9963 DESC | 0.9028 DESC | 0.9513 DESC | 0.8349 DESC
2403 | 1.0000 SOLV | 0.8421 STAL | 0.9992 DESC | 0.9603 DESC | 0.9518 DESC | 0.8305 DESC
2404 | 1.0000 SOLV | 0.9279 DESC | 0.9625 DESC | 0.8094 STAL | 0.9958 DESC | 0.5137 STAL!
2405 | 1.0000 SOLV | 0.7571 STAL | 0.9642 DESC | 0.8310 STAL | 0.9678 DESC | 0.8226 DESC
2406 | 1.0000 SOLV | 0.7969 STAL | 0.9624 DESC | 0.8948 DESC | 0.9805 DESC | 0.8237 STAL!
2407 | 1.0000 SOLV | 0.9348 DESC | 0.9771 DESC | 0.8411 STAL | 0.9505 DESC | 0.7110 STAL
```

## Notes against the pre-registration

- Host: run on kalman, not koopman as the launch line in Amendment 1 says; nothing else in the launch changed.
- Wall time 14.4 h against the estimated 20-25 h (koopman-based).
- Pilots: timing pilot (kalman, s2490, 70 epochs) and extension pilot (koopman, s2492-2493, 900 epochs) are outside the batch
  and not pooled. The checker's accidental two-minute launch at seed 2400 (Amendment 1, incident) left no file and no
  information; the batch's seed-2400 runs are fresh computations.

## Reading for the paper

- The best-rate qualifier "on a ring" is lifted: **on the ring and on the T = 1024 torus, with each step at its own best
  learning rate (four rates 1e-3..3e-2 on the ring, three rates 1e-3..1e-2 on the torus), the |.| step solves 2/8 and 0/8
  against 8/8 for the signed step** (prereg reading wording: "at each parameterisation's best of three learning rates (1e-3
  to 1e-2)").
- On the torus a higher rate does not help |.| (0/8 at each rate; accuracy peaks at 3e-3) and slows the signed step (0/8
  within 900 epochs at 3e-3 and 1e-2, all still descending).
- Not claimed: "at any learning rate", other optimisers or schedules, or the cost at |.|'s rate with decay held at the
  default's exponent (no torus decay control; at 3e-3 the coupled decay is 3x the default's).

## Scope

Torus 64 x 64, T = 1024 (trained and tested at that length), observation-step-free shared rank-4 arms, 1 layer, 2 heads,
d 128, 900 epochs, n = 8, AdamW peak lr 1e-3 / 3e-3 / 1e-2 with weight decay 0.05 coupled to the rate (decay exponent 2.4 /
7.2 / 24.1), cosine with 5% warm-up. "Best rate" = best of the three rates tested; AO's optimum is interior by its loss and
accuracy, SO's is at the grid's lower edge at ceiling. Budget-scoped: 5 of the 8 |.| runs at its best rate were still
descending at 900 epochs.
