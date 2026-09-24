# Review: rank at matched length (RANK_MATCHED_*), 2026-09-23

Scope: `RANK_MATCHED_PREREG.md` (incl. Amendment 1), `run_rank_matched.sh`, `eval_rank_strata.py`,
`analyze_rank_matched.py`, `RANK_MATCHED_RESULTS.md` + JSONs, `RANK_MATCHED_GEOMETRY.md`, and the
supporting code. Everything ran on CPU. Nothing in the repo or in `runs/rank_matched_e900` was
touched. Scripts and outputs are in this directory:

| script | what it checks |
|---|---|
| `analyze_out.txt` | `python3 -m mapformer.analyze_rank_matched`, re-run |
| `check_strata.py` | `kinds()` checked against the env's own cells and revisit mask (400 trajectories) |
| `curves.py` | per-epoch losses, 300-epoch T=1024 batch and the old T=128 batch |
| `stats.py` | t-based MDE, permutation / Wilcoxon / Fisher tests, ANCOVA, within-arm slopes |
| `strata_lossmatched.py` | per-stratum loss-matched contrasts and within-run composition |
| `project_r2.py` (`project_r2.out`) | existence check: trained r=4 projected onto rank 2, loaded into an r=2 model, evaluated |
| `lagbins.py` | finer lag bins for stuck and trained seeds (4 of 6 models finished inside the timeout) |
| `old_geometry.md` | geometry probe on the old T=128 checkpoints, written here (not to the repo) |

**[RUN]** means I verified it by running something. **[READ]** means I judged it from the code or text.

---

## 1. INVALIDATES A CONCLUSION

### 1.1 "The r=4 advantage here is entirely a training-speed difference" does not follow
`RANK_MATCHED_RESULTS.md:25-31` and `:46-52`; `CLAUDE.md:62-63` ("the gap is training speed, not
capability. r=4 trains faster in BOTH batches"); `.claude-memory/project_state.md:39`.

**(a) Loss-matching cannot tell the two hypotheses apart in this design. [READ]** Training and
testing are now the same task. Training loss is revisit cross-entropy at T=1024 on the training map,
and the readout is revisit accuracy at T=1024 on a held-out map. Any in-context solution is
invariant to the map, so accuracy should be about g(loss) for both arms. It is: r = -0.985 within
r=2. So a loss-matched residual near zero is what you expect under "r=4 trains faster", under
"r=4 is more capable", and under "r=4 reaches a lower asymptote". A more capable r=4 would show up
as lower loss, and regressing on loss would remove it. At matched length the residual tests only
one thing: whether the arms turn training-map loss into held-out-map accuracy differently (for
example, map memorisation). That is not the question.

Your suspicion is correct. Loss-matching can never separate the hypotheses here, at convergence
or not. At convergence the final loss IS the capability measure, so you compare it directly and
do not regress it out. The things that can separate them are listed in (d).

**(b) The regression is misspecified, and its support is two r=4 runs. [RUN, `stats.py`]**
- Within-arm slopes differ 3.7x: r=2 -0.196 per log-loss, r=4 -0.053. The pooled -0.066 comes
  from r=4's tail: log-loss spans 6.5 units, and accuracy saturates at 1.000 for losses of both
  0.0015 and 0.012.
- The loss ranges overlap only in 0.209-0.879. Inside that range there are 4 r=2 runs and only
  **2 r=4 runs, both of them failed seeds** (s3 0.671 -> acc 0.715; s4 0.879 -> 0.657).
- So "at equal training loss the ranks are indistinguishable" is an extrapolation from a straight
  line, not a comparison of runs at matched loss.
- ANCOVA (acc ~ arm + log loss) gives arm +0.004, se 0.056, which is the same point estimate and
  far from informative.

**(c) At matched overall loss the arms learned DIFFERENT solutions (exploratory, not
registered). [RUN, `strata_lossmatched.py`]** Overall loss constrains the weighted average over
strata, not the mix, so the mix is not tautological with loss. Within-run gap
`acc(stratum) - acc(plain gap<128)` at T=1024:

| gap to plain gap<128 | r=2 | r=4 | r4 - r2 | MDE | seeds |
|---|---|---|---|---|---|
| plain gap>=128 | -0.092 | -0.002 | +0.090 | 0.083 | 8/8, detectable |
| wrap | -0.203 | -0.017 | +0.185 | 0.149 | 8/8, detectable |

- ANCOVA with overall accuracy as covariate gives the same picture: wrap arm effect +0.211
  (t 4.33); gap>=128 +0.073 (t 2.36); gap<128 -0.028 (t -3.67).
- r=4's accuracy is flat across strata in every seed, including the failed ones: s4 is 0.656 /
  0.657 / 0.662.
- The r=2 runs that learned lose accuracy with distance. s4 (loss 0.209, the best r=2 run): gap
  1-2 0.974, 33-127 0.958, >=128 0.915, wrap 0.632.
- The nearest loss-matched pair is r=2 s4 (0.209) vs r=4 s0 (0.170). Their wrap accuracy is 0.632
  vs 0.902, and gap>=128 is 0.915 vs 0.913.
- This is post hoc and could still be learning order (r=2 learns long range later). But it
  contradicts "the difference between the arms is the loss confound again" (`:62`) and "they are
  not a capability result" (`:46-47`). Those strata are exactly where the arms differ AT matched
  loss.
- It fits the degenerate code r=2 finds on several seeds (both axes on one line, |cos| about
  0.99, i.e. a 1D code of 2D position). Such a code works locally and aliases at long range.

**(d) Representational capability is settled, and r=2 is sufficient. [RUN, `project_r2.py`]**
I took trained T=1024 r=4 checkpoints, projected the rank-4 latent onto its top-2 (uncentred)
singular directions, loaded the result into a `Vanilla` (r=2) model and evaluated it with
`eval_noise_refine.evaluate` (held-out map, seed 1234+s, 100 trials). The CPU r=4 numbers reproduce
`RANK_MATCHED.json` to 4+ decimals.

| r=4 seed (T=1024-trained) | r=4 acc | r=2 projection acc | latent singular values |
|---|---|---|---|
| s6 | 1.0000 | **0.9995** | 7.12, 6.23, 0.12, 0.06 |
| s2 | 1.0000 | 0.9896 | 7.03, 6.43, 1.76, 0.41 |
| s5 | 0.9996 | 0.9537 | 7.49, 6.19, 0.19, 0.06 |
| s1 | 0.9726 | 0.9728 | |
| s0 | 0.9083 | 0.9079 | |

So a 204,373-parameter r=2 model scoring 0.9995 at T=1024 exists. The same holds for the old
T=128-trained r=4: projected to r=2 it scores 0.876 and 0.923 at T=1024 (s0, s1), against
trained r=2's mean of 0.834. Any r=2 deficit, old or new, is about what gradient descent FINDS,
not what r=2 can represent.

**(e) "Trains faster" is itself unsupported. [READ]** At a fixed budget, "lower loss" means
faster OR a lower asymptote. Neither batch separates them. There is also a mild
parameterisation confound in the speed reading (rule 31) **[RUN]**: at init the per-element std of
`w_out` is 0.379 at r=2 and 0.290 at r=4 (`nn.Linear` bound 1/sqrt(r), `model.py:61-62`). The
Delta scale is matched (0.345 vs 0.359), so Adam's relative step on `w_out` is about 1.3x larger at
r=4.

**Defensible sentence.** "Trained and tested at T=1024 for 300 epochs, r=4 reached lower training
loss than r=2 on 7/8 seeds (6/8 vs 0/8 below 0.2, Fisher p=0.007). Held-out accuracy is nearly a
function of that loss (r = -0.985 within r=2), so because train and test are the same task,
loss-matching cannot separate faster training from a better solution. The batch is budget-limited
and leaves the question open. An r=2 model can represent a 0.9995 solution (projection of a
trained r=4), so what is open is learnability, not capacity. Exploratory: at matched loss r=2 is
worse on wrap revisits (8/8 seeds); r=4 is not."

### 1.2 Result 4 misstates the floor comparison
`RANK_MATCHED_RESULTS.md:60-62` [RUN, per-seed read of `RANK_MATCHED_STRATA.json`].

- "r=2 0.563, both above the 0.507 floor" is true only of the arm mean.
- Per seed, r=2 wrap vs its own floor is 0.581/0.506, 0.512/0.503, **0.486/0.511**,
  **0.509/0.512**, 0.630/0.515, 0.776/0.499, 0.510/0.500, **0.496/0.509**.
- So 3/8 seeds are below the floor, 2/8 are within 0.01 of it, and only seeds 0, 4 and 5 are
  clearly above.
- Correct statement: "learnable once trained on" holds for r=4 (6/8 seeds >= 0.84) and for 3/8 r=2
  seeds. The second sentence ("the loss confound again") is contradicted by 1.1(c).

---

## 2. MUST FIX BEFORE THE 8-SEED BATCH (the pilot decision rule is affected now)

### 2.1 The "flat" criterion passes stuck runs, fails solved runs, and loosens with budget
`RANK_MATCHED_PREREG.md:74-77, 109-114`; `analyze_rank_matched.py:46`.

1. **Stuck = flat. [RUN, `curves.py`]** The 4 "flat" r=2 runs (s0, s3, s6, s7; ratios
   0.98-0.99) are exactly the 4 stuck at loss 1.06-1.25 (acc 0.59-0.65, accuracy flat across
   every lag bin). The 4 r=2 runs that learned (s1, s2, s4, s5; ratios 0.92-0.93) are still
   descending. So **no r=2 run is converged in a meaningful sense**, and the results file's
   "r=2 4/8 flat" reads as the opposite. Under the registered gate, 6/8 stuck r=2 seeds would
   PASS convergence and fire R2 "capability". That is rule 10's failure mode exactly.
2. **Solved = not flat. [RUN]** The ratio criterion is relative, and near-zero losses keep
   halving:
   - r=4 s6 is at loss 0.0015 and acc 1.000, yet its ratio is 0.315.
   - The old T=128 batch, which the whole comparison treats as converged (losses
     1e-4 to 0.09, acc 0.993/1.000), scores **flat r=2 1/8, r=4 0/8** under this criterion.
     r=4 ratios there are 0.38-0.79, 250 epochs after the runs solved.

   So at 900 epochs the pilot rule "all 4 pilot runs flat" is likely to FAIL because the r=4 seeds
   SUCCEED. The rule then demands a longer budget, indefinitely.
3. **A fixed 30-epoch window is 3x more lenient at 900 epochs. [RUN]**
   - For L ~ t^-a the last30/prev30 ratio is ((N-15)/(N-45))^-a.
   - It stays within 5% for a <= 0.46 at N=300 but for a <= 1.49 at N=900. Concretely, L ~ 1/t is
     "not flat" (0.895) at 300 and "flat" (0.966) at 900.
   - Under cosine the LR over the 900-run's last 60 epochs is 0.111 -> 0.100 of peak, essentially
     constant at the floor. At 300 it is 0.195 -> 0.100. So the tail is flatter by schedule, not
     by convergence.
4. **Proposed replacement** (register before the pilot is read):
   - SOLVED = mean loss over the last 5% of epochs < eps. Choose eps from the data: final losses of
     0.012 / 0.022 / 0.049 give acc 0.9996 / 1.000 / 0.973, so eps about 0.02.
   - STALLED = not solved, and relative change < 5% between two windows each 5% of the budget
     (scale-free). Stalled runs are search failures and are reported as counts per arm.
   - The capability branches (R2/R3) may not rest on stalled runs. Report success rate
     (solved/8, Fisher exact) separately from accuracy among solved runs.
   - The strongest check is budget invariance: the readout moves by less than MDE/2 from B to 2B.
     The 300 -> 900 pair on seeds 0-1 already gives a partial one, if you are willing to evaluate
     them.

### 2.2 Branches R1/R2/R3 are not well-formed
`RANK_MATCHED_PREREG.md:81-92` [READ; numbers RUN].

- **R1 fires on an underpowered null.** "Primary within MDE" reads anything below about 0.20 as
  "robustness": the 300-epoch MDE is 0.203, 0.237 t-based. That dismisses an effect more than twice
  the +0.085 it is meant to explain away (rule 11). Worse, the noisier the batch (more failed
  seeds, larger sd), the more likely R1 is. Fix: R1 requires MDE < 0.085, or an equivalence test
  with a stated margin; otherwise "unmeasured".
- **R1 and R2 can fire together.** R1's "or both arms >= 0.99 on plain gap<128" clause can hold
  while the primary is > MDE at 7/8, for example when r=4 wins on wrap and long-gap. No precedence
  is stated.
- **Gap and asymmetry.** Primary > MDE with only 6/8 positive fires nothing. R3 has no seed-count
  condition while R2 needs 7/8.
- **R2's meaning.** "r=2 has a deficit at long sequences even when trained on them ... a capability
  claim" is ruled out as a representational claim by 1.1(d). Reword R2 as a learnability/search
  result, and forbid it when r=2's failures are stalled runs (2.1).

### 2.3 `analyze_rank_matched.py` cannot analyse the 900-epoch batch
- **[READ]** `:8-10` hard-code `RANK_MATCHED*.json`; `:44` hard-codes `runs/rank_matched/p0`.
- **[READ]** `:46` hard-codes windows `l[270:]` / `l[240:270]`. On a 900-epoch checkpoint that is
  epochs 241-300, mid-training at LR about 0.9x peak. It would print a meaningless flat count and
  silently re-read the 300-epoch JSONs.
- **Fix:** take TAG / epochs as arguments; use `l[-w:]` vs `l[-2w:-w]` with w scaled to the
  budget; assert `config["epochs"]` matches.
- **The pilot's own readout has no committed script.** The 871-900 vs 841-870 check is to be done
  by hand. Commit it, per the project rule.

### 2.4 Continuing the batch: silent-wrong-result hazards in `run_rank_matched.sh` [READ]
- **SEEDS subset.** Amendment 1 says "run the remaining seeds 2-7" but gives no command. With
  `SEEDS="2 3 4 5 6 7"`, lines 47-53 evaluate only seeds 2-7, producing a 6-seed
  `RANK_MATCHED_e900.json`. It must be run with the default SEEDS (0-7) and rely on the skip at
  `:27`. Write the exact command into the amendment.
- **Stale done markers.** `:46`: the pilot already touches `$REPO/.rank_matched_e900_done` (and
  `$R/.train_done` at `:45`). Any waiter or human checking that marker for the full batch sees
  "done" before it starts. Use a distinct pilot marker, or `rm` both at start.
- **Skip does not check the recipe.** `:27` skips on file existence only. A mistyped TAG with
  EPOCHS=900 reuses the 300-epoch checkpoints, re-evaluates them and rewrites
  `RANK_MATCHED.md/.json/_STRATA.json/_GEOMETRY.md`. Different epochs in the same TAG would mix
  budgets. `train_variant` now saves `epochs`/`n_steps`, so assert they equal EPOCHS/1024 before
  skipping.
- **Silent no-op launch.** `:6`: `flock -n` exits 0 with a message on stdout. A continuation
  launched under `nohup ... > /dev/null` while the pilot driver is still alive (it sits in the
  `:38` loop until its trainers finish) silently does nothing.
- **Code freeze.** Reusing seeds 0-1 means the batch spans two launches about 1.5 h apart. Record
  md5s of `train.py`, `train_variant.py`, `model.py`, `model_rank.py`, `environment.py` and
  `data_parallel.py` now and check them at continuation. "Same seed and code give the same run"
  (`PREREG:111`) is unverified on GPU (no deterministic algorithms), but it is moot as long as
  the runs are reused, not rerun.

### 2.5 The pilot CAN steer the readout
`RANK_MATCHED_PREREG.md:106-107` [READ].

- "Read on training loss only -- so it cannot steer the readout" is false at matched length.
  Training loss is a near-perfect proxy for the primary (r = -0.985), so the pilot shows the
  outcome for seeds 0-1.
- The "proceed" branch is mechanical, but "choose a longer budget" is discretionary. Make it
  mechanical (for example 1800, then 3600, same rule) and symmetric in arms.
- Register the fix in 2.1 before the pilot finishes. At 19:34 it was at epoch 215/900; it will be
  done around 20:40.

### 2.6 Register the readouts the bimodality needs [RUN, `stats.py`]
- **Seeds are bimodal and pairing gives no benefit.** Cross-arm correlation of accuracy by seed is
  -0.18.
- **The MDE is normal-approximation.** The t-based version at n=8 is (t.975,7 + t.80,7) x se =
  0.237, not 0.203. The detection rule |mean| > 2.8 se equals two-sided p about 0.027 at df=7.
  Labelling it "MDE" conflates a critical value with a power statement.
- **The primary has distribution-free p about 0.07-0.15.** On the 300-epoch primary: t = 2.16,
  exact sign-flip permutation p = 0.086, Wilcoxon p = 0.148, sign test 7/8 p = 0.070.
- **The informative contrast is the success rate.** Loss < 0.2 is 0/8 vs 6/8, p = 0.007; 3/8 vs
  6/8 would be p = 0.32.
- **Add as registered readouts:**
  - success rate per arm with Fisher exact;
  - accuracy among solved runs, descriptive;
  - an exact sign-flip test for the paired mean;
  - per-stratum NLL, already registered but not reported (3.1).
- **Consider n=16**, or a design where runs converge reliably (2.7).

### 2.7 Cheaper levers than a blind 3x budget [READ; supporting numbers RUN]
- **Is 900 a reasonable guess?** Under the 300-epoch schedule, r=4 left the plateau (loss < 1.0)
  between epochs 15 and 275, and r4 s4 only at 275. LR stays >= 0.5x peak until epoch 504 at 900
  vs 169 at 300, so 900 plausibly rescues most r=4 seeds. No r=2 run got below 0.2 in 300 epochs,
  and the 4 stuck r=2 runs fell only 0.01-0.03 over epochs 250-300. There is no basis to expect 900 to
  converge r=2, and 2 pilot seeds cannot estimate the stuck fraction.
- **Warm-start from the solved T=128 checkpoints** (`runs/rank_sweep`, same architecture and
  vocab) and fine-tune at T=1024. This is a length curriculum. It removes the from-scratch
  long-context discovery plateau and asks directly whether r=2, given a working map, reaches r=4's
  level at T=1024. That is closer to "capability" than a fixed-budget from-scratch run.
- **Existence plus stability (rule 32).** Warm-start r=2 from the projected r=4 solution in
  1.1(d), trainable, at T=1024. If it holds about 0.99, the r=2 solution is an attractor, and any
  from-scratch deficit is pure search.
- **Larger batch at T=1024.** The stated reason for rejecting it (`RESULTS:76-78`, "changes
  tokens per step away from the T=128 match") is irrelevant to the within-batch r2-vs-r4 contrast.
  The match matters only for comparisons with the old batch.
- **Batch 16 at lr 1e-3** inherits an LR tuned at 128 sequences per step. Tokens per step are
  matched, but the number of independent sequences is 8x smaller, so gradient noise is not. The
  RANK_SWEEP account is that r=2's deficit is an optimisation/conditioning deficit, so a noisier
  regime could penalise r=2 specifically. Untested. A 2-seed lr {3e-4, 1e-3} check would say.

---

## 3. SHOULD FIX

### 3.1 Registered secondaries missing from the results file (rule 30)
- **Per-stratum NLL** is registered (`PREREG:67`) and never reported [RUN]:

  | stratum | r=2 | r=4 | d | MDE | r4 better |
  |---|---|---|---|---|---|
  | gap<128 | 0.637 | 0.260 | -0.377 | 0.622 | 7/8 |
  | gap>=128 | 1.075 | 0.278 | -0.797 | 0.730 | 7/8, detectable |
  | **wrap** | **1.656** | **0.382** | **-1.274** | **0.551** | **8/8, detectable** |

- NLL at T=512 (-0.408) and T=2048 (-0.486), and the new batch's T=2048 strata, are printed by
  `analyze` but absent from `RANK_MATCHED_RESULTS.md`.

### 3.2 "Absent" where the evidence is "unmeasured"
`RANK_MATCHED_RESULTS.md:56-57`.

- The old checkpoints' gap>=128 (+0.062, MDE 0.143) and wrap (-0.003, MDE 0.100) are unmeasured,
  not absent (rule 11).
- The wrap stratum is also below the floor for both arms (0.416/0.413 vs 0.507), so it is a floor
  effect and the difference is uninterpretable, not zero.

### 3.3 "r=4 is still falling steeply on every seed" (`:19-20`)
- s3's ratio is 0.937 (a 6% drop).
- For solved seeds (s6 at 0.0015, s5 at 0.012) the "steep" relative fall is at near-zero loss and
  says nothing about convergence (2.1.2).
- Also say explicitly that the 4 flat r=2 runs are the stuck ones.

### 3.4 Loss-matched model [RUN]
If a loss-matched residual is kept at all, it must:
- use ANCOVA with an arm term (not pooled OLS plus residual differences);
- report the within-arm slopes;
- restrict to the overlap region, or use a model that respects the saturation of accuracy at 1.

`analyze_rank_matched.py:60-62` pairs residuals by seed, which has no meaning here because seeds
do not share loss levels. Use the ANCOVA se instead (0.056, i.e. 2.8 x se = 0.157).

### 3.5 Driver does not check exit codes
`run_rank_matched.sh:3, 47-59` [READ].

- `set -uo pipefail` without `-e`, and no exit-code checks, so `.rank_matched${TAG}_done` is
  touched even if `eval_noise_refine` or the `eval_rank_strata` assert fails.
- The strata JSON is written only after all asserts, so a failure leaves the file missing rather
  than wrong. Still, check `$?` or the artifacts before touching the marker.
- `:38`: the wait regex `/runs\/rank_matched/` matches every TAG's trainers. Harmless but loose.

### 3.6 `eval_rank_strata.py:103-105` assert
- It skips silently when the reference lacks the key. `RANK_SWEEP.json` has no T=2048 entry, so
  the old-checkpoint T=2048 strata were never checked [RUN].
- The 1e-3 tolerance is loose, about 30 targets at T=1024; the observed difference is exactly 0
  [RUN]. Make a missing reference an error (or print "UNCHECKED") and tighten to 1e-9.
- The assert validates trajectory and scoring identity only. It cannot catch a stratification
  bug, because 'all' does not depend on stratum assignment. `check_strata.py` covers that (4.1).

### 3.7 Floor figures quoted inconsistently
- The prereg table's floors (0.515 / 0.496 / 0.506) are seed 0's floors, not the 8-seed means
  (0.506 / 0.500 / 0.507).
- The `eval_rank_strata.py:13` docstring and `CLAUDE.md` quote 0.512.
- None of this changes a verdict. Quote the committed mean.

---

## 4. NOTES (sound, or minor)

1. **Stratification is correct [RUN, `check_strata.py`].**
   - Over 100 trajectories each at T=1024 (two seeds), T=2048 and T=128: zero revisit-mask
     mismatches and zero cell mismatches against the env's own `visited_locations`.
   - The target-to-step mapping `c[i // 2]` is right: target i=2t is the obs of step t.
   - Lag equals t minus the env-true last visit to the wrapped cell. The wrap flag equals "the
     unwrapped position was never visited".
   - Max gap in the <128 stratum is 126. T=128 has zero wraps.
   - Mixed cases, where a plain revisit's most recent visit came via a different unwrapped copy,
     are 0.03% (T=1024) and 0.12% (T=2048). Negligible.
   - The start cell is excluded from `seen` in both the env and `kinds()`. Consistent.
   - The floor (best constant per stratum) is computed correctly. Stratum shares reproduce the
     prereg (0.835 / 0.093 / 0.072).
2. **Both evaluators see identical trajectories [RUN].** Both reseed `np.random` per (model, T);
   corrupt(p=0) and the model draw no numpy RNG; the obs map uses its own RandomState(10000). 'all'
   matches `RANK_MATCHED.json` with difference 0.0 at T=1024 and T=2048, and CPU re-evaluation
   reproduces the GPU accuracies to 4+ decimals.
3. **`analyze_rank_matched.py` reproduces every number in `RANK_MATCHED_RESULTS.md` [RUN].** That
   covers the table, flat counts, loss ranges, ratios, the correlations (-0.985 / -0.836 /
   -0.845), the loss-matched +0.002 (MDE 0.115), the geometry numbers and the old-strata numbers.
   `pair()`, the MDE formula and the counts are implemented as registered. The problems are
   inferential (1.1) and structural (2.3), not arithmetic.
4. **Other numbers verified [RUN].**
   - `CLAUDE.md:59-64` rank numbers: +0.157, MDE 0.203, -0.985, +0.002, 4/8 / 0/8,
     0.495 -> 0.667.
   - Prereg audit claims: old losses non-overlapping (r=2 min 0.0011 > r=4 max 0.0006); old NLL
     means; per-seed old geometry (seeds 1/2/3/6 opposition 0.09-0.14 with |cos| 0.99;
     0/4/5/7 at 0.51-1.22); 94% share (0.835 x 0.096 / 0.085).
   - Design arithmetic: 32,752 vs 32,640 tokens/step, 29,400 steps, 1,470 warmup.
   - Scored targets per batch are about +39% on the held-out map, against the +35% quoted.
   - `CLAUDE.md:128` ("rank ... have NEVER had one") is now stale.
5. **Does matched length remove the robustness confound?** Yes [READ]. Training at T=1024 exposes
   the late-sequence positions, larger accumulated angles and larger distractor counts behind 94%
   of the old effect. Both arms get identical data (same seed gives the same index-seeded stream
   and training map) and identical steps, so nothing is unequal inside the batch.

   What changes is the meaning of the contrast. From-scratch discovery over 2047-token contexts
   at 16 sequences per step is a new optimisation regime, and the rank's known deficit is an
   optimisation deficit. At a fixed budget the contrast measures learnability at long T.
   - The primary is 83.5% gap<128 revisits, so it is the right readout for "does the OLD effect
     survive".
   - It is diluted for the long-range deficit a 1D-aliased r=2 code would predict; that sits in
     the strata registered as non-deciding.
   - Separately [RUN]: in the old batch, T=1024 accuracy does NOT track T=128 training loss within
     r=2 (r = +0.16), so the old effect was not a within-arm loss artefact.
6. **`train_variant.py` config extension is safe [RUN].** No consumer splats `config` into a
   constructor, and `ckpt_guard` reads it as a dict.
7. **`probe_action_geometry.py`: the only caller passes the newly required `--runs-dir`/`--out`
   [RUN grep].**
   - Label mismatch: the column headed |cos(N,E)| computes |cos(N,W)| (`:87`, `pairs[1][0]` is
     action 2 = West). The two have equal magnitude only when E = -W, and differ on the
     non-cancelling seeds.
   - `:95` `orth[-1] if orth else nan` would reuse the previous seed's value in an env with fewer
     than 2 opposite pairs. Harmless on the torus.
   - `:121-128` the verdict ("can be recovered exactly by projecting onto the top two singular
     directions") is hard-coded prose. 1.1(d) shows it is approximately true (0.95-1.00 after
     projection), not exact.
8. **The skip logic is safe with respect to partial checkpoints [READ].** `train_variant` writes
   the `.pt` only after training finishes, so the skip cannot pick up a partial file. TAG
   isolation of run dir, log and outputs works when TAG is set (`_e900` gives
   `runs/rank_matched_e900`, `rank_matched_e900.log`, `RANK_MATCHED_e900{.md,.json,_STRATA.json,_GEOMETRY.md}`),
   and `RANK_SWEEP_STRATA.json` is correctly skipped. The GPU picker (less-loaded device, own
   `python3` trainers only, 4.5 GiB free) is sound.
