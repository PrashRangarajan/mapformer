# RANKRECIPE: does shared rank 2 still fail the T = 1024 torus at the best of three recipes? -- pre-registration

Registered 2026-10-24 before any batch run; amended by Amendment 1 (the independent check), folded in below. Origin: the
user's decision from the drafting session's big-picture review (attack A2; the review file `BIG_PICTURE_2026-10-24.md` lives on
the drafting machine and was not in this repository at registration -- Amendment 1 finding 10), relayed by that session: shelve RANKDECAY (parked on branch
`rankdecay-wip`; its independent check found omega has no function of its own) and test whether "rank D+1 is needed" is
really "rank D is harder at the default recipe". Every rank-2 failure on record was at one recipe (lr 1e-3, wd 0.05): RANK_MI
0/8, RANKCLOCK 1/8. Code: `analyze_rankclock`-based `analyze_rankrecipe.py` (selftest 10 checks), `run_rankrecipe.sh`; selftest 16 checks after Amendment 1; no
training-code change (main's `train.py` is untouched). **This file is md5-guarded and re-checked before every launch: it is
final at launch.**

## 1. Cells

The RANKWD / RANK_MI recipe verbatim except lr and weight decay: torus 64 x 64, T = 1024, 900 epochs, cosine (5% warm-up),
98 batches of 16, AdamW, 1 layer, 2 heads, d 128, no landmarks, 3 data workers, shared rank. Fresh seeds **2600-2607**
(unused: RANKDECAY never launched). 32 runs, GPU 1 only, two at a time, ~17 h.

| cell | model | lr | weight decay | role |
|---|---|---|---|---|
| r2_R0 | Vanilla (shared rank 2) | 1e-3 | 0.05 | the default recipe; the in-batch rank-2 anchor (rule 12) |
| r2_R1 | Vanilla | 3e-3 (pilot rule: stable, Sec. 5) | 0.05 | a higher learning rate |
| r2_R2 | Vanilla | 1e-3 | 0.2 | stronger decay (RANKWD: decay is what rank 4 needs) |
| r4_R0 | Vanilla_r4mi (shared rank 4) | 1e-3 | 0.05 | the comparison: rank 4 at the default |

Rank 4 gets one recipe on purpose: the search favours rank 2, so a rank cost that SURVIVES it is conservative. **That is
liberal for the other direction (Amendment 1 finding 1):** if rank 2's best recipe closes the gap, rank 4 was never run at that
recipe, so a residual rank cost at matched recipes is not excluded -- hence the CLOSES label names the recipe and says "rank 4
not searched". Within a seed, all rank-2 cells share every initial weight with each other and with rank 4's non-bottleneck
weights (RANK_MI).

**The grid is not a pure learning-rate / decay search (finding 7).** AdamW's decoupled decay scales with lr x wd, so the
total decay exponent wd * sum(lr) over the schedule is 2.41 at R0, **7.24 at R1** (lr 3e-3 triples it) and **9.66 at R2**:
both alternatives are higher-decay recipes, there is no lower-decay arm and no lr + wd combination, and R1 sits at the grid
edge (unbracketed, unlike LRSWEEP's interior optimum). The claim is "the best of these three recipes", not rank 2's own best
recipe.

## 2. Existence (rule 14)

A rank-2 solution exists and is held: projected rank-2 models score 0.9955 frozen at T = 1024 and are held under training
(`RANK_PROJ_RESULTS.md`); RANKCLOCK's r2 construction scored 1.0000 frozen and held a 60-epoch warm start at lr 1e-3, wd 0.05
(`RANKCLOCK_PREREG.md` Sec. 3). Cheap sanity check at the two new recipes (the same construction, 60 cosine epochs):

Run 2026-10-24 on GPU 1 (`runs/rankrecipe_pilot/warm_lr3e-3`, `warm_wd0.2`), source RANKCLOCK's r2 construction
(`runs/rankclock_pilot/warmstart/r2/Vanilla.pt`, frozen 1.0000), scored frozen with `eval_noise_refine.evaluate` on fresh maps
at T = 1024 (60 trials):

| warm start (60 cosine epochs) | after | loss first / max / final-5% | class | |
|---|---|---|---|---|
| lr 3e-3, wd 0.05 | 0.9998 (nll 0.0004) | 0.0912 / 0.5649 / 0.0342 | SOLVED | **HELD** |
| lr 1e-3, wd 0.2 | 1.0000 (nll 0.0001) | 0.0119 / 0.2925 / 0.0100 | SOLVED | **HELD** |

Both new recipes hold a rank-2 solution, so a rank-2 failure at either is a failure to find it. The 3e-3 recipe perturbs the
solution harder (peak loss 0.565) and ends close to the SOLVED line (0.034 < 0.05). **Limit (finding 8):** 60 epochs at wd 0.2
reach a decay exponent of ~0.64, ~7% of the batch's 9.66, so this checks holding under the new optimiser, not under the full
decay the batch applies; the warm start also keeps a solution at the scale trained under wd 0.05 (as in RANKWD).

## 3. Registered analysis (`analyze_rankrecipe.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); non-finite loss = DIVERGED, never SOLVED. Each run's lr,
weight decay, recipe and len(losses) = 900 are asserted; a missing evaluation row raises.

- **G0** r4_R0 SOLVED >= 6/8, else UNREADABLE.
- **G1** r2_R0 SOLVED <= 2/8, else RANK-2 DIFFICULTY NOT REPLICATED.
- **B (primary): rank 2 at its best recipe vs rank 4 at R0**, using `analyze_lrsweep`'s `select`, `b1_test`, `b1_decide`
  (imported): rank 2's recipe is selected on SOLVED, then the lowest mean `sel_score` (final-5% training loss), then the earlier
  column (R0, R1, R2) -- never on held-out accuracy (a selftest gives the loss-worse recipe the higher accuracy and checks it is
  not chosen); rank 4's single recipe is replicated across the three columns. **Null (finding 3):** that construction gives
  "selection" to both relabelled groups while the data gives it only to rank 2, so the selection-aware p is miscentred and
  is NOT a valid test alone; in every scenario the reviewer tried it was <= the naive p. The decision p is max(selection-aware,
  naive at the selected recipe), so in practice **the naive p decides** -- valid, and conservative for SURVIVES because selection
  favours rank 2. Unpaired tests ignore the within-seed pairing (conservative). Route test as `analyze_lrsweep.label_from`:
  rank 4 better if (decision p_SOLVED < .05 with k_r4 > k_best) or (decision p_acc < .05 **and** d = acc_best - acc_r4 <= -0.05)
  (finding 2: the accuracy route has the house effect-size floor; no Holm across the two routes, as in LRSWEEP).
  - k_best <= 2, k_r4 >= 6 and rank 4 better -> **RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH**
  - k_best >= 6 at R1 or R2, that recipe beats rank 2's own default (Fisher p < .025: Bonferroni x2 for choosing between the two
    alternatives), and BOTH decision p-values against rank 4 >= .05 -> **RANK-D COST CLOSES AT <recipe> (rank 4 not searched)**
    (finding 1: the closing needs inference both ways and has no effect-size escape)
  - otherwise -> **UNMEASURED** -- including, by design, k_best 3-5 and, e.g., rank 2 best 3/8 against rank 4 8/8 (Fisher p 0.026:
    a gap, but not the registered survival branch; finding 13)
  B is computed and printed (as descriptive) even when G0 or G1 fails (finding 9).
- **Also printed:** per cell SOLVED, accuracy at T = 512 / 1024 / 2048 (DIVERGED runs count accuracy 0, as G-4), windowed
  classes (SOLVED / STALLED / DESCENDING / RISING / DIVERGED), UNSTABLE count, mean selection score; the selected rank-2 recipe;
  the effective per-move angle |omega (W_out W_in e_move)| growth and its max raw and wrapped (finding 6: invariant under both
  step-map gauges but not under a 2 pi wrap per move -- a magnitude, not the function; moves only, though in Vanilla observation
  tokens step too; descriptive and wrapped in try/except so a bad snapshot cannot block the verdict); r(log10 final loss, acc)
  per length (rule 2).
- **UNSTABLE (finding 5):** G-4's clause (`runaway`) on unsolved, NON-DIVERGED runs, with its step term replaced by the
  gauge-invariant max effective angle (the pre-omega step moved 1.32 -> 6.44 under a function-preserving gauge change). Validated
  on runs of known status (rule 3): on 24 stored runs -- RANKCLOCK's 8 rank-2 runs and RANKWD's 16 rank-4 runs, 14 of them
  unsolved -- it flags none; every run's highest post-warm-up loss stays at least 0.47 below its H_map (medians 0.58-0.82 below).

**Power.** With rank 4 at 8/8: rank 2 best 0 / 1 / 2 gives naive Fisher p 0.0002 / 0.0014 / 0.0070 (the naive p decides; see the
null note). With rank 4 at 6/8, rank 2 best 2/8 gives Fisher p 0.13, which survives only through the accuracy route
(p < .05 and d <= -0.05). CLOSES needs the best recipe at >= 6/8 against rank 2's default: 6 vs 0 p 0.007, 6 vs 1 p 0.041
(< .025 fails), 7 vs 1 p 0.010.

**Budget (rules 3, 4).** "Within 900 epochs". A rank-2 recipe that solves later is outside this claim; the windowed classes
are printed beside the reading so a DESCENDING-heavy cell is visible.

## 4. Predictions

None registered as expected; the readings are fixed above. SURVIVES strengthens the paper's rank claim (the cost holds
against a three-recipe search that favours rank 2); CLOSES rewords it to "rank D is harder at the default recipe", scoped to the
named recipe with rank 4 unsearched there (an r4 cell at that recipe would remove the caveat; the reviewer's alternative,
not run here).

## 5. Pilot and the R1 rule (seeds 2692-2693, outside the batch, never pooled)

Rule (from the drafting session): if the pilot shows lr 3e-3 is unstable for rank 2 at T = 1024, R1 becomes 2e-3 and the reason
is recorded here; the batch stays at 32 runs. "Unstable" = G-4's registered UNSTABLE clause on either pilot run (any epoch after
warm-up above that seed's H_map + 0.05, a final move step above 1e3, or a final-5% loss above 1.9). Pilot seeds 2690-2691 were
used by the shelved RANKDECAY pilots, so this pilot uses 2692-2693.

Read with the committed reader (`python3 -m mapformer.analyze_rankrecipe --pilot`, the amended UNSTABLE clause), 2026-10-24
04:00; GPU 1, four runs sharing it, 900 epochs:

| pilot | recipe | class | final-5% loss | UNSTABLE | max post-warm-up loss (H_map) | max effective angle |
|---|---|---|---|---|---|---|
| lr3e-3_s2692 | lr 3e-3, wd 0.05 | DESCENDING | 0.3861 | no | 1.199 (2.084) | 5.51 |
| lr3e-3_s2693 | lr 3e-3, wd 0.05 | STALLED | 0.4066 | no | 1.432 (2.078) | 2.87 |
| wd0.2_s2692 | lr 1e-3, wd 0.2 | DESCENDING | 0.4819 | no | 1.497 (2.084) | 0.40 |
| wd0.2_s2693 | lr 1e-3, wd 0.2 | STALLED | 0.5636 | no | 1.598 (2.078) | 3.03 |

**R1 rule: neither lr-3e-3 run is UNSTABLE, so R1 = 3e-3 as registered** (no change to `analyze_rankrecipe.RECIPES`). None of the
four pilots solved; two seeds per recipe, outside the batch, never pooled -- an anecdote that changes nothing registered.
Timing: 8.5 s/epoch with four runs sharing GPU 1; the batch runs two at a time (~4.2 s/epoch each), ~17 h for 32 runs.

## Amendment 1 (independent results-blind code check, rule 29)

Reviewer read this file, the analyzer and everything it imports, the driver and its precedents, train.py / train_variant.py
(unchanged on main since RANKWD's 36f1752), the models and evaluator, and the baseline results, blind to `runs/` and logs; ran the
selftest and scratch scenarios (selection, accuracy route, gauge, args). Found the recipe plumbing, the eval / analyzer layout,
the selection (training loss only; the tie test is real), h_map's seed and the 45-epoch warm-up, and the power and budget
figures correct. Findings and dispositions:

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | MAJOR | the "harder at the default" branch had no inference (6/8 vs its own default 2/8 is p 0.13), fired with rank 4 significantly better, and is the LIBERAL side of an unsearched rank 4 | FIXED: renamed RANK-D COST CLOSES AT <recipe> (rank 4 not searched); needs best vs r2_R0 Fisher p < .025 and both decision p's vs rank 4 >= .05; selftest cases for each failure mode. The alternative (an r4 cell at the best recipe) is not run |
| 2 | MAJOR | the accuracy route had no direction or effect-size floor, so 2/8 vs 6/8 could read a survival on a tiny accuracy gap; the parenthetical was false | FIXED: d <= -0.05 on the accuracy route (the house rule); parenthetical rewritten; selftest cases for a large and a small significant gap |
| 3 | MINOR | the selection-aware null is miscentred (both groups "selected"); in practice the naive p decides | ACCEPTED and stated in Sec. 3: valid and conservative for SURVIVES; not a test alone |
| 4 | MINOR | an unfailable selftest check; tautological recipe asserts; missing cases | FIXED: the check removed; cases added for the accuracy route, DIVERGED, UNSTABLE, a wrong weight decay and a missing eval row (16 checks). The recipe plumbing was checked by the reviewer through the real parser |
| 5 | MINOR | UNSTABLE's step term is gauge-dependent; DIVERGED runs could be flagged; DIVERGED accuracy not forced to 0; threshold unvalidated on Vanilla; no committed pilot reader | FIXED: gauge-invariant max effective angle; `not solved and not div`; DIVERGED accuracy 0; validated on 24 stored runs (none flagged); `analyze_rankrecipe --pilot` committed and used for the R1 decision |
| 6 | MINOR | the angle readout is a magnitude (not wrap-invariant) and moves only | FIXED: wrapped max printed; caveats printed and in Sec. 3 |
| 7 | MINOR | R1 also triples total decay; both alternatives are higher-decay; R1 unbracketed | ACCEPTED: Sec. 1 states the decay exponents (2.41 / 7.24 / 9.66); title and labels say "three recipes" |
| 8 | MINOR | the wd-0.2 warm start reaches ~7% of the batch's decay exponent | recorded as a limit in Sec. 2 |
| 9 | MINOR | B suppressed when a gate fails; geometry could block the verdict | FIXED: B always computed and printed (descriptive when a gate fails); geometry in try/except |
| 10 | MINOR | the cited review file is not in the repository | FIXED: citation says it lives on the drafting machine |
| 11 | NOTE | driver checks out (recipes from RECIPES, layout, md5 per launch, counts, lock, GPU 1) | -- |
| 12 | NOTE | launch would be refused by the user's two unrelated local changes; pilot root must not be runs/rankrecipe | the two changes set aside in a named stash for the batch; the pilot root is runs/rankrecipe_pilot |
| 13 | NOTE | 3/8 vs 8/8 (p 0.026) reads UNMEASURED | stated in Sec. 3 |
| 14 | NOTE | power and budget figures correct | -- |
