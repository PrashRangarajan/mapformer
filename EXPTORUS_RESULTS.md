# Do the exp and softplus phase increments pay the sign learnability cost in attention on the T = 1024 torus? -- results (EXPTORUS, registered, `EXPTORUS_PREREG.md` + Amendments 1-2)

Batch 2026-10-26 03:42 -> 13:28 (9 h 46 min) on **kalman** GPUs 0-1 (2 runs per GPU), code `782ba4f` (committed, clean;
md5 guard over 106 files, `runs/exptorus/code_md5.txt`, re-checked before every launch and `md5sum -c` clean afterwards),
torch 2.10.0+cu128, 64 launches for 64 runs (no relaunch, no failure in `exptorus.log`), marker `.exptorus_done`. Every
launch recorded no foreign trainer and no other user on the GPUs (`runs/exptorus/launch_occupancy.txt`). Fresh seeds
3400-3407. Independent results-blind check before launch (`docs/audits/2026-10-26/exptorus_audit.md`: no BLOCKER, 2 MAJOR,
5 MINOR, all folded into Amendment 1). Pilot on outside seeds 3490-3491 (`runs/exptorus_pilot/PILOT_ANALYSIS.txt`,
Amendment 2; never pooled). Analysis `analyze_exptorus.py` -> `EXPTORUS_ANALYSIS.txt`, `EXPTORUS_VERDICTS.json`; evaluations
`runs/exptorus/eval/` (held-out map, env seed 10000, 100 walks at T = 1024 and 2048; revisit strata).

**Task and recipe** = TORUS_LRSWEEP (G-4) unchanged: 64 x 64 torus, 16 object types, p_empty 0.5, T = 1024 moves, batch 16,
98 batches per epoch, 900 epochs, cosine (5% warm-up, floor 0.1 x lr), AdamW wd 0.05 coupled to lr (decay exponent
sum(lr_t) x wd 2.4 at 1e-3, 7.2 at 3e-3), 1 layer, 2 heads, d 128, `--fast-attn`. Observation-step-free shared rank-4 class:
only the four move tokens step. **Arms** differ only in the function on the step readout s = W_out W_in e(move): signed
(s; anchor), |.| (|s|; anchor gate), **exp** (exp(s), the LRU-style positivity; primary), **softplus** (softplus(s),
Mamba-Delta-style; primary). At a seed every arm and rate starts from the same weights and sees the same training stream.
Each arm at lr 1e-3 and 3e-3: 4 arms x 2 rates x 8 seeds. **Floors** (T = 1024 eval stream, post hoc,
`docs/audits/2026-10-20/WHAT_WHERE_T1024.md`): blank 0.506, best n-gram 0.575, retrace 0.843.

**Reading (registered): BOTH PAY.** In one-layer attention on the T = 1024 torus, each arm at its own best of two learning
rates, the exp increment solves **0/8** (best rate 1e-3, accuracy 0.916) and the softplus increment **0/8** (best rate
3e-3, accuracy 0.908) against the signed increment's **8/8** (best rate 1e-3, accuracy 0.993): SIGNED LEARNS BETTER for
both (Holm over the two primaries: SOLVED p 0.0003 each). The exp no-cost seen in the SSM on the ring (G-3 A2) does not
transfer to attention on the torus. Qualifiers: exp budget-scoped (6/8 runs at its best rate still descending); softplus
budget-scoped (5/8 descending), its best rate is the grid's top rate (optimum not bracketed; "best" = best tested), and its
accuracy route does not fire (selection-aware p 0.11): the SOLVED route carries it. Anchor gate PASS (|.| 0/8 vs signed 8/8).

> **Disclosure (Amendment 2, verbatim).** "**Disclosure: the batch is now partly pilot-informed for exp.** The pilot is not
> blind to exp's behaviour on the torus: 0 of 4 exp runs solved; both 3e-3 runs are UNSTABLE (loss above H_map + 0.05 after
> warm-up); at 1e-3 neither is UNSTABLE, one stalled and one still descending, and their cancelling channels cancel mostly
> by a wrap, mostly from the past-pi start set. Nothing in the design, analysis, thresholds or labels changes; the
> registered readings and the Sec. 4 predictions stand as written before the pilot. The results file will carry this
> disclosure beside the exp readings, and the pilot's exp numbers are never pooled with the batch."

## B1 (registered primaries and anchor gate): best vs best

Selection on training loss only: most SOLVED of 8, then lowest mean selection score (final-5% loss capped at the first-epoch
loss), then the lower rate. Test: selection-aware permutation null over all 12,870 splits of the 16 per-seed rate profiles
(selection redone in each split); decision p per route = max(selection-aware, naive) (Fisher on SOLVED, unpaired
permutation on accuracy); Holm over exp and softplus per route; label rule = ROBUST's cell rule on the Holm p.

| arm | best rate (why) | SOLVED | accuracy T = 1024 | d vs signed | selection-aware p (SOLVED / acc) | Holm decision p (SOLVED / acc) | label |
|---|---|---|---|---|---|---|---|
| signed (anchor) | 1e-3 (8/8 vs 0/8 at 3e-3) | **8/8** | **0.9926** | -- | -- | -- | -- |
| \|.\| (anchor gate; raw p) | 1e-3 (0/8 at both; selection score 0.529 vs 0.539) | **0/8** | 0.8581 | -0.1345 | 0.0002 / 0.0002 | (raw) 0.0002 / 0.0002 | SIGNED LEARNS BETTER: gate **PASS** |
| **exp** | 1e-3 (0/8 at both; 0.311 vs 0.797) | **0/8** | 0.9159 | -0.0767 | 0.0002 / 0.0003 | **0.0003 / 0.0006** | **SIGNED LEARNS BETTER** |
| **softplus** | 3e-3 (0/8 at both; 0.369 vs 0.492) | **0/8** | 0.9078 | -0.0848 | 0.0002 / 0.1100 | **0.0003** / 0.1100 | **SIGNED LEARNS BETTER** (SOLVED route) |

Naive at the selected rates: exp Fisher 0.0002, permutation 0.0003; softplus Fisher 0.0002, permutation 0.0005. The
softplus accuracy route is the one place where the selection-aware null is much wider than the naive one (0.11 vs 0.0005);
the label rests on SOLVED. DESCENDING at the selected rate: |.| 1, exp 6, softplus 5. UNSTABLE and unsolved at the selected
rate: |.| 0, exp 1 (below the >= 2 that would mark the cell as optimiser instability), softplus 0.

**Registered readings** (`arm_reading`, `batch_reading`):
- exp: **"EXP PAYS THE SIGN COST IN ATTENTION ON THE TORUS (SIGNED LEARNS BETTER at best rates) [6 exp run(s) still
  DESCENDING at its best rate: budget-scoped]"**. Matched-rate 1e-3 per-rate label (Amendment 1.2): SIGNED LEARNS BETTER
  (exp's selected rate is 1e-3, so the matched and selected cells coincide). *Partly pilot-informed (disclosure above).*
- softplus: **"SOFTPLUS PAYS THE SIGN COST IN ATTENTION ON THE TORUS (SIGNED LEARNS BETTER at best rates) [softplus's best
  rate is the grid's top rate: its optimum is not bracketed; 'best' = best TESTED] [5 softplus run(s) still DESCENDING at
  its best rate: budget-scoped]"**. Matched-rate 1e-3 per-rate label: SIGNED LEARNS BETTER.
- Batch: **"BOTH PAY: in attention on the torus the exp (LRU-style positivity) and softplus (Mamba-Delta-style)
  content-dependent increments pay the cost (the SSM's exp no-cost does not transfer)"**.

## Per rate (declared secondary; each arm vs signed at the same rate; Holm over the 2 rates within arm and route)

| lr | signed | \|.\| | exp | softplus |
|---|---|---|---|---|
| 1e-3 | **8/8, 0.9926** (8 SOLVED) | 0/8, 0.8581 (7 STAL, 1 DESC) | 0/8, 0.9159 (2 STAL, 6 DESC; 1 UNSTABLE) | 0/8, 0.8489 (5 STAL, 3 DESC) |
| 3e-3 | 0/8, 0.9639 (8 DESC) | 0/8, 0.8648 (5 STAL, 3 DESC) | 0/8, 0.7429 (6 STAL, 2 DESC; **6 UNSTABLE**) | 0/8, 0.9078 (2 STAL, 5 DESC, 1 RISING) |
| accuracy T = 2048 (1e-3 / 3e-3) | 0.980 / 0.900 | 0.733 / 0.797 | 0.763 / 0.665 | 0.741 / 0.823 |

| arm, lr | d vs signed | Fisher p (Holm) | perm p (Holm) | label |
|---|---|---|---|---|
| \|.\| 1e-3 | -0.1345 | 0.0002 (0.0003) | 0.0002 (0.0003) | SIGNED LEARNS BETTER |
| exp 1e-3 (= G-3 A2's rate) | -0.0767 | 0.0002 (0.0003) | 0.0003 (0.0003) | SIGNED LEARNS BETTER |
| softplus 1e-3 | -0.1437 | 0.0002 (0.0003) | 0.0003 (0.0006) | SIGNED LEARNS BETTER |
| \|.\|, exp, softplus 3e-3 | -0.099, -0.221, -0.056 | 1.0 (1.0) | 0.0002, 0.0002, 0.0073 | UNREADABLE (signed 0/8 < 4) -- not a sign contrast: signed is at its known-bad torus rate |

At A2's rate (1e-3), with no rate selection on either side, exp and softplus solve 0/8 against signed 8/8.

**UNSTABLE** (RUNAWAY = a post-warm-up epoch above the seed's context-free loss H_map + 0.05, or a final move step > 1e3;
COLLAPSED = final-5% loss > 1.9; DIVERGED = non-finite loss). 7 of 64 runs, all exp, all RUNAWAY by loss; none diverged,
none COLLAPSED, no step above 1e3 (largest 35.1); no other arm has an UNSTABLE run.

| run | max post-warm-up loss - H_map (epoch; H_map) | final-5% loss | step max | class |
|---|---|---|---|---|
| exp 1e-3 s3405 | +0.138 (67; 2.134) | 0.431 | 13.5 | STALLED |
| exp 3e-3 s3400 | +0.075 (60; 2.089) | 0.515 | 24.4 | DESCENDING |
| exp 3e-3 s3402 | +0.897 (121; 2.098) | 1.085 | 11.3 | STALLED |
| exp 3e-3 s3403 | +0.542 (71; 2.060) | 0.699 | 17.8 | STALLED |
| exp 3e-3 s3405 | +0.859 (150; 2.134) | 0.656 | 35.1 | STALLED |
| exp 3e-3 s3406 | +0.401 (84; 2.046) | 0.975 | 19.3 | STALLED |
| exp 3e-3 s3407 | +0.629 (274; 2.087) | 1.363 | 14.2 | STALLED |

exp at 3e-3 is an unstable cell (6/8); it was not selected (exp's best rate is 1e-3) and no learnability sentence rests on it.

## Declared secondaries (no verdict depends on them)

- **Threshold sensitivity** (SOLVED at loss 0.1, selection redone; raw decision p): |.|, exp and softplus each 0/8 vs 8/8,
  SOLVED p 0.0002 (accuracy 0.0002 / 0.0003 / 0.11) -> SIGNED LEARNS BETTER for all three.
- **Accuracy at T = 2048** (robustness, not capability) at the selected rates: |.| 0.733, exp 0.763, softplus 0.823 vs signed
  0.980; selection-aware p 0.0002 / 0.0002 / 0.020.
- **Cross-fit** (rates chosen on one half of the seeds, tested on the other; n = 4, where Fisher reaches p < .05 only at 4/4
  vs 0/4): exp chooses 1e-3 in both halves, tested 0/4 vs 4/4 in both (0.904 vs 1.000; 0.928 vs 0.985), Fisher 0.029;
  softplus chooses 3e-3 in both halves, 0/4 vs 4/4 in both (0.880 vs 1.000; 0.936 vs 0.985), Fisher 0.029; |.| chooses
  3e-3 then 1e-3, 0/4 vs 4/4 both times.
- **Revisit strata at T = 1024** (mean over seeds; plain lag < 128 | plain lag >= 128 | wrap-only; always-blank ~0.51):

| lr | signed | \|.\| | exp | softplus |
|---|---|---|---|---|
| 1e-3 | 0.992 / 0.993 / 0.996 | 0.912 / 0.651 / 0.522 | 0.977 / 0.695 / 0.524 | 0.888 / 0.724 / 0.577 |
| 3e-3 | 1.000 / 0.943 / 0.609 | 0.930 / 0.589 / 0.498 | 0.800 / 0.474 / 0.454 | 0.971 / 0.659 / 0.529 |

  At their selected rates exp and softplus keep short-gap revisits (0.977, 0.971) and sit near the blank level on wrap-only
  revisits (0.524, 0.529), the |.| pattern; their accuracy above retrace (0.843) comes from short gaps.
- **Usage-weighted cancelling share / opposite-sign share** (wrap-aware; channels with |wrap(first-move angle)| > 0.05;
  opposite sign read on angles wrapped to (-pi, pi]): signed 1.00 / 1.00 at both rates; |.| 0.19 / 0.09 (1e-3), 0.53 / 0.17
  (3e-3); **exp 0.81 / 0.26 (1e-3, selected)**, 0.38 / 0.08 (3e-3); softplus 0.76 / 0.21 (1e-3), **0.89 / 0.41 (3e-3,
  selected)**. The channels attention uses mostly cancel in exp and softplus models, yet none solves: descriptive, no account
  read from it.
- **Mechanism census** (per (pair, channel) cell over 8 seeds, pairs N/S and W/E; non-blind |wrap(first-move angle)| > 0.05;
  cancelling |wrap(pair sum)| < 0.1; by a wrap = round(sum / 2 pi) != 0, through zero = 0; past-pi start set measured from
  each run's reconstructed initial weights):

| lr, arm | non-blind | cancelling | by wrap | through zero | cancelling from the past-pi start set | P(cancel \| past-pi start) vs P(cancel \| not) | turns | omega end / init |
|---|---|---|---|---|---|---|---|---|
| 1e-3 signed | 693 | 693 | 0 | 693 | 36 | 1.00 vs 1.00 | 0 | 0.22 |
| 1e-3 \|.\| | 439 | 46 | 25 | 21 | 17 | 0.31 vs 0.08 | 0-2 | 0.13 |
| **1e-3 exp (selected)** | 443 | **107** | **97** | 10 | **95** | 0.31 vs 0.09 | 0-2 | 0.11 |
| 1e-3 softplus | 427 | 98 | 77 | 21 | 81 | 0.38 vs 0.08 | 0-4 | 0.12 |
| 3e-3 signed | 237 | 237 | 0 | 237 | 17 | 1.00 vs 1.00 | 0 | 0.05 |
| 3e-3 \|.\| | 271 | 46 | 32 | 14 | 4 | 0.20 vs 0.17 | 0-2 | 0.03 |
| 3e-3 exp | 377 | 39 | 19 | 20 | 19 | 0.12 vs 0.09 | 0-1 | 0.05 |
| **3e-3 softplus (selected)** | 159 | **64** | **57** | 7 | **46** | 0.68 vs 0.20 | 0-2 | 0.02 |

  Initial past-pi record (mean over seeds, of 64 channels; per pair N/S, W/E, both): signed 2.6 / 2.8 / 1.0; |.| 2.9 / 4.4 /
  1.4; **exp 22.6 / 22.0 / 20.2** (analytic 22: exp(0) = 1 and 2 omega > pi on 11 channels per head, **34%**); softplus 15.1
  / 15.5 / 13.9 (analytic 16, 25%). G-3's SSM ring ladder (grid 32) put exp past pi on 13 of 32 channels per head (**41%**):
  the torus start puts a smaller share past pi.
  Read: at its selected rate exp cancels where A2's exp did, by a wrap (97 of 107) and mostly on channels that started past
  pi (95 of 107; a past-pi channel cancels 3.4x as often as another), the pattern the prereg predicted "if exp solves any
  run" -- here it does so without solving: on the torus the past-pi start yields wrap cancellation on a minority of
  non-blind cells (107 of 443) and no map within 900 epochs. Softplus at 3e-3 shows the same pattern (57 of 64 by wrap, 46
  from the past-pi set). Weight decay shrinks omega to 0.11 (exp, 1e-3) and 0.02 (softplus, 3e-3) of its initial value
  (Amendment 1.6: under decay the exp / softplus steps drift toward the uniform clock 1 / ln 2).
- **Rule 2:** r(log10 final loss, accuracy) = -0.618 at T = 1024 and -0.757 at T = 2048 over the 64 runs. The primaries fire
  on SOLVED (0/8 vs 8/8), which does not depend on held-out accuracy.

## Cross-batch references (labelled, never pooled; rule 12)

| batch | mixer, task | exp | softplus | \|.\| | signed |
|---|---|---|---|---|---|
| **EXPTORUS (this)** | attention, torus T = 1024, best of 1e-3 / 3e-3 | **0/8** (1e-3), 0.916 | **0/8** (3e-3), 0.908 | 0/8 (1e-3), 0.858 | 8/8 (1e-3), 0.993 |
| G-3 A2 (`G3_SSM_A2_RESULTS.md`, seeds 2100-2107) | one-layer diagonal complex SSM, ring 32, T = 128, lr 1e-3 | **6/8, 0.985: NO SIGN COST** | 0/8, 0.938 | 0/8, 0.756 | 7/8, 0.994 |
| G-3 block A (`G3_SSM_RESULTS.md`) | SSM, ring, lr 3e-3 | 0/8, 8/8 RUNAWAY | 1/8 | 0/8 | 8/8 |
| SIGNROUTE (`SIGNROUTE_RESULTS.md`, seeds 1800-1807) | attention, ring 32, fixed rates 1e-3 / 3e-3 | 1/8 / 1/8 | 1/8 / 3/8 | 0/8 / 2/8 | 8/8 / 8/8 |
| TORUS_LRSWEEP G-4 (`TORUS_LRSWEEP_RESULTS.md`, seeds 2400-2407, same recipe) | attention, torus, best of 1e-3..1e-2 | -- | -- | 0/8 (3e-3), 0.890 | 8/8 (1e-3), 1.000 |
| SIGN_MATCHED (`SIGN_MATCHED_RESULTS.md`, seeds 0-7) | attention, torus, full model (observations step), lr 1e-3 | -- | 1/8, 0.781 | 0/8, 0.821 | 8/8, 0.998 |

This batch's anchors replicate G-4 on fresh seeds: signed 8/8 at 1e-3 and 0/8 (all descending) at 3e-3; |.| 0/8 at both
rates (|.|'s selected rate moves from 3e-3 to 1e-3 by the loss tie-break, 0.529 vs 0.539). One signed run (s3402) is SOLVED
on loss but scores 0.942 held out (G-4's signed runs at 1e-3 were all 1.000).

## Predictions (prereg Sec. 7, written before the pilot) against outcomes

| prediction | outcome |
|---|---|
| anchor gate SIGNED LEARNS BETTER (~0.95) | PASS: 0/8 vs 8/8 |
| signed selects 1e-3 (~0.9) | 1e-3: 8/8 vs 0/8 (all DESCENDING) at 3e-3 |
| \|.\| 0-1/8 at both rates | 0/8 at both |
| exp: SIGNED LEARNS BETTER ~0.6, UNMEASURED ~0.25, NO SIGN COST ~0.15 | SIGNED LEARNS BETTER (0/8; partly pilot-informed) |
| softplus: SIGNED LEARNS BETTER ~0.75 | SIGNED LEARNS BETTER (0/8; SOLVED route only) |
| exp UNSTABLE at 3e-3 (>= 2 runs) ~0.3 | yes, 6/8 (the pilot, run after the prediction, had shown 2/2) |
| batch BOTH PAY ~0.5, MIXED / UNMEASURED ~0.3, EXP ESCAPES ~0.15 | BOTH PAY |
| census, if exp solves any run: cancelling channels mostly from the past-pi set, by a wrap | exp solved none (condition not met); descriptively its cancelling channels are mostly past-pi-start (95/107) and by a wrap (97/107) |

## Checks on the reading

- **Code provenance:** `md5sum -c runs/exptorus/code_md5.txt` from the repo root: all 106 files OK after the batch.
- **`experiment_audit`** (rule 3), `python3 -m mapformer.experiment_audit --runs-dir mapformer/runs/exptorus` from
  /home/prashr (read-only; also per rate dir): exits 2 -- no (json, pt) pairs in its layout, so it runs check [1] only. Its
  windowed classes agree with the registered per-seed classes in every cell: 1e-3 signed 8 SOLVED, |.| 7 STALLED + 1
  DESCENDING, exp 2 + 6, softplus 5 + 3; 3e-3 signed 8 DESCENDING, |.| 5 + 3, exp 6 + 2, softplus 2 + 5 + 1 RISING. Its
  "not all arms converged" flag concerns the non-solving arms and signed at 3e-3; the qualifiers above state it.
- **Numbers re-derived** from the evaluation JSONs (scratch script, not committed): every cell mean at T = 1024 and 2048
  matches the analysis to the printed digits; exact permutation p for exp 1e-3, softplus 1e-3 / 3e-3 and |.| 1e-3 vs signed
  1e-3 recomputed over all 12,870 splits (0.0003 / 0.0003 / 0.0005 / 0.0002), equal to the naive p printed.
- **Power (prereg Sec. 5):** with signed at 8/8 the SOLVED route fires for an arm at <= 3/8 (at 3/8 it would fall to Holm
  0.051 had the other primary been 8/8); 0/8 is the strongest reachable outcome for both primaries.
- **Evaluation mode** under-reports one-layer runs below ceiling (attention-dropout scale, `DROPOUT_RESCORE.md`); no re-score
  was declared; all arms are read the same way, and SOLVED does not depend on it.

## Per seed (accuracy at T = 1024 and class: SOLV SOLVED, STAL STALLED, DESC DESCENDING, RISI RISING; ! = UNSTABLE)

```
seed | 1e-3:SO     | 1e-3:AO     | 1e-3:EXP    | 1e-3:SPL    | 3e-3:SO     | 3e-3:AO     | 3e-3:EXP     | 3e-3:SPL
3400 | 0.9988 SOLV | 0.8863 STAL | 0.9274 DESC | 0.7067 STAL | 0.9778 DESC | 0.9363 DESC | 0.8626 DESC! | 0.9207 DESC
3401 | 1.0000 SOLV | 0.9064 STAL | 0.9530 DESC | 0.8102 STAL | 0.9588 DESC | 0.8553 STAL | 0.8046 STAL  | 0.8879 STAL
3402 | 0.9420 SOLV | 0.8790 STAL | 0.9092 STAL | 0.9255 DESC | 0.9595 DESC | 0.7783 STAL | 0.5429 STAL! | 0.9837 DESC
3403 | 1.0000 SOLV | 0.7671 STAL | 0.9231 DESC | 0.9221 DESC | 0.9493 DESC | 0.8987 DESC | 0.8510 STAL! | 0.9501 DESC
3404 | 1.0000 SOLV | 0.8243 STAL | 0.8757 DESC | 0.6904 STAL | 0.9567 DESC | 0.8957 DESC | 0.8651 DESC  | 0.8680 RISI
3405 | 1.0000 SOLV | 0.8395 STAL | 0.8792 STAL!| 0.9104 DESC | 0.9769 DESC | 0.8328 STAL | 0.7891 STAL! | 0.8775 STAL
3406 | 1.0000 SOLV | 0.9158 DESC | 0.9213 DESC | 0.8743 STAL | 0.9672 DESC | 0.8633 STAL | 0.6247 STAL! | 0.9410 DESC
3407 | 1.0000 SOLV | 0.8465 STAL | 0.9380 DESC | 0.9517 STAL | 0.9645 DESC | 0.8583 STAL | 0.6036 STAL! | 0.8336 DESC
```

## Reading for the paper

- **The sign sentence now covers the exp and softplus increments in attention on the headline task:** in one-layer attention
  on the T = 1024 torus, each at its best of two learning rates, exp (LRU-style positivity) and softplus (Mamba-Delta-style)
  content-dependent increments solve 0/8 against the signed increment's 8/8 (Holm p 0.0003 on SOLVED), as |.| does. Wording
  (Amendment 1.7): "the exp and softplus content-dependent increments", not "the LRU and Mamba parameterisations" (the LRU's
  phase is input-independent; Mamba's Delta multiplies the state matrix). Budget-scoped (6/8 and 5/8 still descending);
  softplus "at its best tested rate" (not bracketed).
- **exp escapes in one place only:** the one-layer SSM on the ring at lr 1e-3 (G-3 A2, 6/8 vs 7/8). It pays in attention on
  the ring (SIGNROUTE, 1/8 at both rates) and in attention on the torus (this batch, 0/8 at both rates). "The cost is shown
  for |s| and softplus(s), not exp" becomes "|s|, softplus(s) and exp(s) pay in attention; exp escapes only in the SSM on the
  ring at 1e-3".
- The CRITICAL_REVIEW 2026-10-26 B-B2 gap ("exp has never been run in attention on the headline task") is closed.

## Scope -- what this separates and what it does not

- **Separated:** whether exp's SSM escape is a property of the exp parameterisation alone. It is not: the same function on
  the same readout, in attention on the torus, pays (0/8 at both rates; matched-rate 1e-3 cell 0/8 vs 8/8).
- **Not separated:** why exp escapes in the SSM. Between G-3 A2 and this batch the mixer (unnormalised linear kernel vs
  softmax), the task (ring 32, T = 128, 300 epochs vs torus 64 x 64, T = 1024, 900 epochs) and the start geometry (exp past
  pi on 41% of channels on the SSM's grid-32 ladder vs 34% on the torus's grid-64 ladder; and the torus needs a channel to
  cancel on two pairs) all change at once; the batches are never pooled. SIGNROUTE (attention, ring: exp 1/8) points at the
  mixer and/or start, not the task alone, but is a third batch with its own recipe. This batch cannot say which of mixer,
  task and start explains the SSM escape. The census shows exp's cancelling cells on the torus are the past-pi-start ones
  cancelling by a wrap, as in the SSM -- the start is used, but on the torus it does not reach a map.
- One layer, 2 heads, d 128, observation-step-free shared rank 4, one task and length, two rates, 900 epochs, n = 8, AdamW
  wd 0.05 coupled to the rate (decay exponent 2.4 / 7.2; under decay the exp / softplus steps drift toward the uniform clock
  1 / ln 2, Amendment 1.6), so BOTH PAY means these two increments under this recipe, not non-negativity alone. exp at 3e-3
  is an unstable cell (6/8 RUNAWAY) and licenses nothing. No SSM on the torus and no exp started below pi were run.

## Notes against the pre-registration

- Ran as registered (option B, `EXPTORUS_RATES = 1e-3 3e-3`); 9.8 h wall time against 9.6 h projected by the pilot.
- No other job and no other user on the GPUs at any launch; RNN_TORUS started after the marker (13:30).
- The exp readings are partly pilot-informed (Amendment 2, disclosure above); the pilot's four exp runs (seeds 3490-3491) are
  not pooled.
