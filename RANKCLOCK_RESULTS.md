# Does extra rank help because it gives the clock an exit? -- results (registered, `RANKCLOCK_PREREG.md` + Amendment 1)

Batch 2026-10-21 18:48 -> 10-23 02:16 on GPU 1 only, two jobs at a time, code `a33d6d0` (clean tree, md5 guard over 96 files),
torch 2.10.0+cu128, 56 runs: 7 arms x fresh seeds 1900-1907, torus 64 x 64, T = 1024, 900 epochs, the RANK_MI recipe.
Independent results-blind check before launch (Amendment 1: 3 MAJOR, 5 MINOR, 2 NOTE, all folded in). Existence checked
frozen AND trainable before launch: every class contains a solution and holds it (prereg Sec. 3). Analysis
`analyze_rankclock.py` -> `RANKCLOCK_ANALYSIS.txt`, `RANKCLOCK_VERDICTS.json`; accuracy `RANKCLOCK.md` / `.json`,
`RANKCLOCK_WD0.md` / `.json`. Pilots (seeds 1990-1992) are in the prereg Sec. 7 and are not pooled.

**Reading (registered primary): PARTIAL -- the per-axis clock slot solves 3/8 against rank 2's 1/8 (Fisher p 0.57).** The
registered bar for CLOCK EXITS HELP was >= 6/8 with p < .05; the bar for CLOCK EXITS DO NOT HELP was <= 2/8. Neither fired.

**Registered secondary E2: THE CLOCK IS THE OBSTACLE.** Drift-forbidden rank 2 solves **8/8** (accuracy 1.000 on every seed,
like rank 4) against actions-only rank 2's 1/8 (Fisher p 0.0014), and actions-only is no better than rank 2 (1/8 vs 1/8). At
the same rank that fails, removing the clock by construction is enough; removing the observation steps is not.

Together: the clock is what stops rank 2, but handing it a separate exit at rank 2 does not reliably let training remove it
within 900 epochs. "Extra rank helps BECAUSE it gives the clock an exit" is therefore **not supported** at the registered
bar -- and not refuted (PARTIAL; see Budget).

## Gates

- **G0** rank 4 SOLVED 8/8 (>= 6): the recipe reproduces RANK_MI's 8/8 on this host. Pass.
- **G1** rank 2 SOLVED 1/8 (<= 2): the rank-2 difficulty replicates (RANK_MI 0/8). Pass.

## All arms

| arm | step map | SOLVED | classes (S / T / D) | acc T = 512 | acc T = 1024 | acc T = 2048 |
|---|---|---|---|---|---|---|
| r2 | shared rank 2 | 1/8 | 1 / 2 / 5 | 0.954 +/- 0.042 | 0.918 +/- 0.053 | 0.851 +/- 0.056 |
| **slotxy** | r2 + per-axis clock slots (primary) | **3/8** | 3 / 1 / 4 | 0.984 +/- 0.037 | 0.972 +/- 0.046 | 0.927 +/- 0.059 |
| slot | r2 + per-token clock slot (P2) | 3/8 | 3 / 2 / 3 | 0.979 +/- 0.023 | 0.951 +/- 0.045 | 0.891 +/- 0.075 |
| **DF** | r2, drift forbidden, observation steps 0 | **8/8** | 8 / 0 / 0 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.001 |
| AO | r2, observation steps 0, moves free | 1/8 | 1 / 0 / 7 | 0.990 +/- 0.011 | 0.962 +/- 0.029 | 0.890 +/- 0.072 |
| r4 | shared rank 4 | 8/8 | 8 / 0 / 0 | 1.000 +/- 0.000 | 1.000 +/- 0.001 | 0.986 +/- 0.018 |
| r4wd0 | shared rank 4, weight decay 0 | 1/8 | 1 / 6 / 1 | 0.766 +/- 0.128 | 0.760 +/- 0.130 | 0.748 +/- 0.130 |

SOLVED = final-5% training loss < 0.05 (`stats_core.classify_run`); classes S SOLVED, T STALLED, D DESCENDING; no run diverged.
Accuracy: `eval_noise_refine`, fresh maps (env seed 10000), 100 trials, mean +/- sd over 8 seeds.

## Registered readings

| readout | label | numbers |
|---|---|---|
| **P** (per-axis slot vs r2) | **PARTIAL** | 3/8 vs 1/8, Fisher p 0.5692; acc 0.972 vs 0.918, permutation p 0.0500 |
| P2 (per-token slot vs r2) | PARTIAL | 3/8 vs 1/8, Fisher p 0.5692; acc 0.951 vs 0.918, permutation p 0.2068 |
| **E2** (G1+) | **THE CLOCK IS THE OBSTACLE** | DF 8/8 vs AO 1/8, Fisher p 0.0014; AO vs r2 1/8 vs 1/8, p 1.0000; DF vs r4 8/8 vs 8/8 (descriptive) |
| X2 (r4 at wd 0: park or remove?) | X2 UNREADABLE | r4wd0 SOLVED 1/8 < the 4 required |
| X2b (P4 ordering) | NO ORDER | r4wd0: no usable runs (1 solved); r4: read clock first in 2/4 usable runs, sign p 1.0000 |

The slots' permutation p of 0.0500 on accuracy is printed beside P as registered and is not a decision rule: the primary is
on SOLVED. The slots learned small angles (median max |omega * slot| 0.142 / 0.124 for clock_x / clock_y, 0.068 for the
per-token slot), so they were used, not left at zero.

## Budget (rules 3, 4) -- read before quoting the primary

The slot arms are not stalled at rank 2's level; most are still moving: slotxy has 4 DESCENDING and 1 STALLED of its 5
unsolved runs, slot 3 DESCENDING and 2 STALLED. Rank 2 itself has 5 of 7 unsolved runs DESCENDING. The registered claim is
"within 900 epochs"; a slot that solves later than rank 4 (or a rank 2 that would solve with more epochs) is consistent with
this table. PARTIAL here is a speed-or-capability question this batch does not separate. For reference: rank 2 did NOT solve
in 1800 epochs from scratch (0/8 vs rank 4 7/8, `LOOP_RANK_E1800_P1_RESULTS.md`), so a longer budget is not certain to rescue
the slot arms either.

## Not registered, flagged as post hoc

- **Weight decay looks close to necessary for rank 4 here.** Rank 4 at weight decay 0 solves 1/8 against 8/8 at 0.05 (same
  table as DF vs AO, Fisher p 0.0014), with 6/8 STALLED, and is the only arm whose T = 512 accuracy is far from the ceiling
  (0.766). No contrast between r4 and r4wd0 was registered -- the wd-0 arm was there to read X2 -- so this is a post-hoc
  observation to confirm in a registered batch, not a result. It is also why X2 could not be read.
- **In the solved wd-0.05 rank-4 runs the clock is small at the end**: median shrink 0.134 of its initial size, readgain
  0.026 of initial, u_ratio 0.022 (8/8 solved). With decay on, removal and decay of a parked clock cannot be told apart (the
  reason X2 needed the wd-0 arm), so this says only that the end state carries little clock.
- **DF holds its solution past training length**: 1.000 +/- 0.001 at T = 2048, above rank 4's 0.986 +/- 0.018.
- Geometry medians at the end of training (all runs, descriptive): u_ratio r2 0.169, slotxy 0.243, slot 0.056, AO 0.481, r4
  0.022; shrink (|u_end| / |u_init|) r2 1.86, slotxy 3.15, slot 0.41, AO 7.30, r4 0.13. In r2, slotxy and AO the latent clock
  GREW over training (shrink > 1), most in AO; in the per-token slot arm it shrank (0.41), as it did in rank 4.

## Checks on the reading

- **Rule 2:** r(log10 final loss, acc) = -0.537 (T = 512), -0.655 (1024), -0.824 (2048) over 56 runs. The registered readings
  are on SOLVED; accuracy is descriptive.
- **Rule 14 (prereg Sec. 3):** every class, including both slot classes, contains a frozen solution (1.0000) and held it
  through a 60-epoch warm start at peak lr 1e-3, so the slot arms' failures are search failures, not capacity or an
  optimiser artefact of their 12.9 rad-per-step sensitivity.
- **Within-batch:** all 56 arms in one batch, one GPU, one code version; r2, slot, slotxy, DF and AO share every initial
  weight at a seed (bitwise, `test_rankclock.py`).

## What this changes

- **Rank 2's obstacle is the clock.** Forbid it and the same rank-2 map solves 8/8 and generalises to 2T; remove only the
  observation steps and nothing changes. This is the cleanest rank result so far and it is registered.
- **The "clock exit" mechanism for why EXTRA rank helps is not supported** as registered: two different explicit exits at
  rank 2 reach 3/8, not rank 4's 8/8. The theory's parking story (`RANK_THEORY.md`) should be stated as consistent with end
  states but not shown causal; the RANK_THEORY wording constraint ("rank above D gives the drift a map-free exit" only after
  G1) now has its answer: not at the registered bar.
- **Open, and cheap to test:** whether weight decay is necessary for rank 4 (post hoc 8/8 vs 1/8 here), and whether the
  slots are slow rather than unable (a longer budget for slotxy and r2 together).
