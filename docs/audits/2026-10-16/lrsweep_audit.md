# LRSWEEP independent code check (rule 29), 2026-10-16, before launch

Scope: `LRSWEEP_PREREG.md`, `analyze_lrsweep.py`, `run_lrsweep.sh` (commit 9005a48; HEAD is now 63aa45d, which changes only
`docs/paper/`, so the code is identical) and the pilot script `scratchpad/lrsweep_pilot.sh` (read as it is on disk now: DRV_MAXPG
changed from 6 to 4, otherwise as written). Precedents: `ROBUST_PREREG.md` + Amendment 1, `analyze_robust.py`, `run_robust.sh`,
`robust_audit.md`. Blind to results. No LRSWEEP batch result exists. I did not create `runs/lrsweep/` or
`runs/lrsweep_pilot/`; the second now exists because the pilot was launched elsewhere during this check, and I did not read
it. No GPU used. All CPU runs were in the scratchpad (`lrsweep_check/`) on seeds 1580-1583, except the analysis-crash test,
which reused the author's synthetic smoke tree (fake seeds 2000-2007). Check scripts and their outputs are copied to
`docs/audits/2026-10-16/lrsweep_check/`. No code edited.

No BLOCKER. The code does what the pre-registration says. Every cell's command line sets the registered lr, weight decay,
deltas, epochs and seeds. The 1e-3 cell is the ROBUST def cell. lr reaches AdamW and the schedule at every rate. Divergence is
caught, kept and cannot crash anything. The selection rule, the 12,870-split null and the reading match independent
reimplementations. There are two MAJOR points, both about the design:
- the reading's COST CLOSES / NO COST split depends on the signed arm's loss tie-break between rates where it is at ceiling;
- if B1 reads SIGNED LEARNS BETTER, the high-rate cells confound learning rate with the weight decay coupled to it, and that
  decay acts on the angle scale.

## Verified (no action)

1. **Command lines.** Dry run (`DRV_DRYRUN=1 DRV_GPUS="0 1"` in a `git clone --shared` of HEAD): 64 launches in the order
   seed, then rate, then arm; no duplicate `--output-dir`. Each line is `train_gauge --arm {SO,AO} --deltas=1,-1 --seed
   1500..1507 --lr {1e-3,3e-3,1e-2,3e-2} --weight-decay 0.05`. Epochs (300), batches (98), batch (128), T (128), size (32),
   data workers (3) and trials (400) are `train_gauge` defaults, as in ROBUST, and `analyze_lrsweep.run_record` asserts all
   of them except `n_trials` (MINOR 7). The md5 guard covers 96 files (94 imported modules + 2 driver files). Then it fails
   closed: MISSING x 128, "FAILED: missing runs -- done marker NOT set".
2. **The 1e-3 cell is ROBUST def.**
   - The `ring()` line at L=1e-3 is byte-identical to `run_robust.sh`'s `ring()` line at C=def apart from the output
     directory. The 3e-3 line is likewise ROBUST lrx3.
   - All 94 modules shared by the two guards are md5-identical to `runs/robust/code_md5.txt` (the code ROBUST's ring runs
     trained with). LRSWEEP adds `analyze_lrsweep.py` and `run_lrsweep.sh`; ROBUST's extra files were the torus evaluators.
   - Bitwise on CPU (`bit3.sh`: seed 1581, 3 epochs x 4 batches, both arms, `OMP_NUM_THREADS=1`): identical losses, all 24
     state-dict tensors `torch.equal`, identical `eval.json`.
   - Caveat on the check method, not the code: with `OMP_NUM_THREADS=2` two sequential runs of one command on CPU are NOT
     bitwise equal (epoch 2 differs at ~1e-7; `bit2.sh`). At 1 thread they are. This is CPU multi-threaded reduction. The GPU
     path is the one the batch uses, and the ROBUST audit showed it bitwise over 300 epochs against GAUGE.
3. **lr reaches AdamW and the schedule; nothing clamps it** (`lr_check.sh` with AdamW instrumented, seed 1580, 20 epochs x 5
   batches, both arms, at 1e-3, 3e-3, 1e-2, 3e-2 and the extension rates 1e-1, 3e-1; `lr_summary.txt`). There is one param
   group of 24 tensors (204,243 parameters; every parameter, including `path_integrator.omega`, `action_to_lie.w_in/w_out`
   and the LayerNorm gains), with weight decay 0.05. Per rate: a 5% linear warmup from lr/20, a peak exactly equal to lr
   (ratio 1.000000), cosine down to 0.1002 x lr at the last step, and the saved args record lr and wd.
4. **Divergence is caught, kept and cannot crash the driver or the analysis.**
   - Real NaN divergence on CPU at lr 1e2 / 1e3 / 1e4 (seed 1582, 6 epochs): `train_gauge` completes and writes `<arm>.pt`
     (NaN losses, 24/24 tensors NaN) and `eval.json` (acc 0.0, nll NaN, wrap [NaN, NaN]). So the driver sees `eval.json` and
     neither retries nor fails.
   - Analysis: five real NaN-weight checkpoints (3 AO, 2 SO at 3e-2) dropped into the author's synthetic "unmeasured" tree
     (`nantree/`). `run_record` classes them DIVERGED, keeps them and scores them 0 accuracy; `opp_share`,
     `usage_weights`, `wrap_share_weighted` and `nanmean` return NaN without raising. Exit code 0. Verdict JSON and all
     secondaries are written.
5. **Selection rule** (most SOLVED, then lower mean selection score, then lower rate). An independent loop implementation
   over 3000 random groups with ties forced: the only mismatches are means that are equal in exact arithmetic and differ by
   float rounding. The module treats them as ties and takes the lower rate, as registered: the `SOLVED x 1e6` key absorbs
   differences below ~1e-9. When SOLVED = 0 at every rate the key is exact, so a 1e-17 rounding difference decides instead
   of the rate. This cannot matter with real losses. Held-out accuracy never enters: `select()` receives only `solved` and
   `score`. `sel_score` is as registered: the final 5% (15 epochs at 300), capped at the epoch-1 loss, and the epoch-1 loss
   when the tail is non-finite.
6. **Selection-aware null.**
   - Brute-force reimplementation (every split, selection redone by the loop rule) on 60 random datasets at n = 4: p-values
     and selected rates identical. At n = 8 on a random dataset: identical (0.451748 / 0.949650).
   - Enumeration: `_relabellings(16, 8)` gives 12,870 rows, all distinct, exact (not Monte Carlo), observed split included
     once. The complement construction is correct.
   - With one rate the null equals Fisher (selftest).
   - Two-sided p by `|T| >= |T_obs|`, with a 1e-12 tolerance on the accuracy route only (the SOLVED statistic is an integer).
7. **Reading.**
   - An independent implementation from the prereg text agrees with the module on all 205 combinations (5 labels x every
     (AO, SO) rate pair, on the 4-rate grid and on the extended 5-rate grid): 0 mismatches.
   - `--selftest` passes. Every B1 label and every reading is reached, plus the edge qualifier, and `label_from` equals
     `analyze_robust.cell_label` (Amendment 1 form, including the SO >= 6/8 condition for NO SIGN COST).
8. **Power sentence (observed counts)** checked with `--power` and three worst-case configurations: SO 8/8 at every rate and
   AO 3/8 at the best rate, on the same, rotating or disjoint seeds across rates. Whenever SO is at ceiling the
   selection-aware p is <= Fisher, so the decision p equals Fisher at the selected rates: AO 3/8 gives 0.0256 (fires) and
   4/8 gives 0.077. The statement is correct as written. MINOR 3 states it in true rates.
9. **Size under the profile null** (`sim_null.py`, 4000 simulations each; profiles iid, with a per-profile effect, or with a
   shared seed effect).
   - Selection-aware p alone: SOLVED route 0.001-0.022 (discrete, conservative); accuracy route 0.048-0.051, i.e. exact as
     claimed. The shared seed effect makes it conservative (0.001 / 0.016).
   - Decision p = max: SOLVED route <= 0.005; accuracy route <= 0.022. SIGNED LEARNS BETTER fires 0.1-0.9%.
10. **Driver** (`drvtest.sh`, in the clone, dry run):
    - Resume skips runs with `eval.json` ("skip ... (done)") and runs in flight (a dummy process carrying the `--output-dir`
      gives "skip ... (running)"): 61 launches.
    - A dirty tracked file is REFUSED before any wait; a held lock is REFUSED; a committed change to `train.py` after the
      first guard record gives "ABORT: training code changed".
    - An extension amendment (RATES += 1e-1, committed) passes the tag check: 80 launches, NEXP 160, and the analysis gets
      `--rates 1e-3,3e-3,1e-2,3e-2,1e-1`, so the edge qualifier uses the extended top rate.
    - It sources `lib_driver.sh`, takes the `flock -n` lock, checks the clean tree and the committed files before the first
      `drv_wait_slot`, allows up to 6 trainers per GPU counted over every `mapformer.train_*` of the user, uses the rule-23
      awk everywhere, re-checks md5 before the analysis, and writes the artifacts before the marker.
11. **Seeds and streams.** Batch seeds are `(s * 1000003 + i) mod 2^31`. For s <= 1599 and i < 29,400 there is no modular
    wrap (1,599,034,197 < 2^31), and seeds 1 apart are 1,000,003 apart, so streams cannot overlap. `find runs -name
    '*s15[0-9][0-9]*'` finds only the pilot's 1590-1591 now. At a seed, initial weights and stream do not depend on lr: both
    are built before `train()`; ROBUST audit item 7, and epoch-1 losses agree across command variants here.
12. **The extension rule is fixed before data and does not peek.**
    - It is committed in 9005a48 before any pilot and decided on outside seeds (1590-1591). The selection-aware null is
      conditional on the grid, so adding 1e-1 changes power, not validity.
    - With 0/2 vs 0/2 the mean selection score decides, so loss noise triggers the extension about half the time. That is
      generous to AO and costs 16 runs. Acceptable.

## Findings

### MAJOR 1 -- COST CLOSES vs NO COST hinges on the signed arm's loss tie-break between its ceiling rates
`reading()` splits NO SIGN COST / NON-NEGATIVE LEARNS BETTER into COST CLOSES AT HIGHER LR (`r_ao > r_so`) and NO COST AT
BEST RATES (`r_ao <= r_so`). The prereg glosses the second as "the cost does not replicate at the best rates even without a
rate shift (that would contradict ROBUST def and E3 at 1e-3)". The expected regime is SO at 8/8 at several rates (ROBUST: 8/8
at 1e-3 and 3e-3). There `r_so` is set by the second tie-break, the lowest mean final-5% loss among those rates. With a
higher peak and floor (0.1 x lr), a higher rate can easily reach the lowest final loss. That is an arbitrary property of the
ceiling, not the rate at which signed steps start to work. Demonstration through the module's own functions (AO identical in
both cases: 1/8 at 1e-3, 4/8 at 3e-3, 8/8 at 1e-2; SO 8/8 at 1e-3..1e-2):
- SO final loss lowest at 1e-2 gives NO SIGN COST and "NO COST AT BEST RATES (AO best 0.01, SO best 0.01)";
- SO final loss lowest at 3e-3 gives NO SIGN COST and "COST CLOSES AT HIGHER LR (AO best 0.01 > SO best 0.003)".

In the first case AO solves 1/8 at the shared default, yet the prescribed text would say the 1e-3 cost "does not replicate"
and withdraw the title "pending an account".
**Fix:** compare AO's best rate with the LOWEST rate at which SO reaches its best SOLVED count:
`r_so_min = min{r : k_SO(r) = k_SO(r_so)}`.
- COST CLOSES AT HIGHER LR if B1 is NSC or NLB and `r_ao > r_so_min`.
- NO COST AT BEST RATES if `r_ao <= r_so_min`, i.e. AO matches SO without needing a larger rate.
- The gloss "contradicts ROBUST def / E3" applies only when the per-rate 1e-3 secondary also reads NO SIGN COST; otherwise
  say so.

`r_so` (the tie-broken rate) stays the rate at which SO's accuracy enters B1. This is a design change to `reading()`, so it
is for the author, with a selftest case for each branch.

### MAJOR 2 (it matters only if B1 reads SIGNED LEARNS BETTER) -- the high-rate cells confound lr with decay that acts on the angle scale
AdamW decays every parameter by lr_t x wd per step. Over the run the decay exponent is sum lr_t x wd ~ 0.05 x lr x 29,400 x
0.55 ~ 805 x lr:

| rate | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|
| decay exponent | 0.8 | 2.4 | 8 | 24 | 80 |
| time constant 1/(lr x wd), steps | 20,000 | 6,700 | 2,000 | 670 | 200 |

At 1e-3 the run never reaches AdamW's weight-norm equilibrium; ROBUST wd0 ~ def is consistent with that. From 1e-2 up, decay
is fully in force, and it acts on `omega`, `w_out` and the LayerNorm gains, i.e. on the angle scale (all decayed: item 3).
MECHANISM found that AO learns the map by wrapping (move-angle sums > pi). So decay that pulls angles toward zero is a
plausible AO-specific handicap at exactly the rates where AO's best is expected. The prereg discloses the coupling. But under
SLB the reading says "the scope gains 'at each parameterisation's best of four learning rates'; W1 is answered". A reviewer
would answer: "at your high rates you decayed AO's angles away." Under NSC / NLB / UNMEASURED the confound works against the
finding and needs no fix.
**Fix (choose one before launch):**
- (a) Add a decay-matched control to the same batch (rule 12): SO and AO at 1e-2 with wd 0.005 and at 3e-2 with wd
  1.666667e-3 (lr x wd = 5e-5, the default's), seeds 1500-1507. That is 32 runs, ~2.7 GPU-h (~1.4 h wall on two GPUs at 6
  per GPU); the batch becomes 96 runs, ~4 h wall. Declared secondary D, read per rate with the ROBUST rule (Holm over its 2
  rates). Under SLB, if D reads NO SIGN COST at AO's selected rate, the reading carries "[decay-coupled: at matched lr x wd
  the cost is not present at <rate>]". `run_record` must accept the D cells' wd, and the driver needs the cells and tags.
- (b) No runs: the SLB reading's scope sentence becomes "at each parameterisation's best of four learning rates, with AdamW
  weight decay 0.05 coupled to the learning rate (lr x wd from 5e-5 to 1.5e-3)", and W1 is described as "answered for the
  coupled recipe".

I recommend (a) if the GPU budget allows: it is the only cell that can defend an SLB against the obvious objection.

### MINOR 1 -- best-of-4 by SOLVED favours |.| by more than "fairly": AO's best count is inflated by selection
SO is expected at ceiling, where selection cannot inflate it. AO picks the best of 4 noisy counts. Simulated (`sim3.py`, 2000
runs each, SO 1.0 at every rate):

| AO's true per-rate solve probability | AO@best mean count (true x 8) | NO SIGN COST fires |
|---|---|---|
| 0.6 at three rates | 5.99 (4.8) | 13% |
| 0.75 at two rates | 6.71 (6.0) | 33% |
| 0.9 at two rates | 7.64 (7.2) | 78% |

So a COST CLOSES reading ("learn the map at their own rate as reliably as signed steps at theirs") occurs in a third of the
cases where AO's true best-rate solve probability is 0.75 against SO's 1.0. The bias works against the paper's claim, so it
is conservative for the title. But the COST CLOSES / NO COST wording would overclaim reliability.
**Fix (wording; no change to the rule):** COST CLOSES / NO COST state the comparison as "the cost is not detectable at AO's
best tested rate (AO k/8 at r_ao vs SO k/8; AO@best is selection-inflated)". They also report the cross-fit, which is
selection-free: the two directions pooled give AO and SO SOLVED out of 8, each seed scored at the rate chosen on the other
half. A gate on the cross-fit is NOT recommended. Simulated, it cuts false NSC only from 33% to 21% and also drops true NSC
(AO 1.0 at its best) from 100% to 80%.

### MINOR 2 -- the null's validity statement is too strong in one place and the max rule's justification is incomplete
- "Under the null (arm labels exchangeable per profile) this is exact whatever the selection does." It is exact under
  exchangeability of all 16 profiles, which requires equal distributions at EVERY rate (the profile null) and no shared seed
  effect.
  - The pairing (shared init and stream at a seed) breaks full exchangeability. Simulated with a shared seed effect, the
    unpaired test is conservative (SOLVED 0.001, accuracy 0.016 at alpha 0.05). Valid in that direction, as the prereg
    says, but it is not "exact".
  - B1's own null, best-vs-best equal, is composite: profiles may differ at non-best rates. Under composite nulls the
    selection-aware p alone is anti-conservative. Example (C1 in `sim2.py`): SO at 0.6 on four rates vs AO at 0.6 on one
    rate gives SOLVED-route rejection 0.088.
- That is the real reason the max rule is needed. With it, SLB fires 5.2% in that adversarial case (two-sided alpha 0.05;
  both routes) and 1.5-2.1% in C2-C4. Also, in the expected regime (SO at ceiling) the selection-aware p is <= Fisher, so the
  decision p IS the naive Fisher on AO's inflated count. In the ROBUST-like case, AO (0.125, 0.5, 0.5, 0.3), the SOLVED-route
  power is 0.11 (decision) vs 0.42 (selection-aware p alone).

**Fix (text):** replace "exact whatever the selection does" with "exact under exchangeability of the 16 profiles (equal
distributions at every rate); conservative under a shared seed effect (simulated); B1's best-vs-best null is composite, and
under composite nulls the selection-aware p alone can be anti-conservative (simulated 0.088 at alpha 0.05), so the decision
uses max(selection-aware, naive) (simulated SLB rate <= 5.2% in an adversarial composite null). With SO at ceiling the
decision p equals Fisher at the selected rates." Optional declared secondary: the paired selection-aware test (within-seed
swaps, 2^8 = 256 relabellings, minimum p 0.0078), which matches the pairing. Simulated: label-level power 0.34 vs 0.25 for the
registered route; but it is less protected in composite nulls: 6.5% on the SOLVED route and 9.7% at label level in C1
(no max rule). So keep it descriptive.

### MINOR 3 -- state the power in true rates, so a non-SLB is not over-read
The power statement is in observed counts. In true per-seed solve probability at AO's best rate (SO 1.0; `sim_null.py` /
`sim2.py`), SLB fires 25% at 0.5 (ROBUST's 4/8 at 3e-3), 67% at 0.3, 89% at 0.2 and 99% at 0.1. **Fix:** one sentence in the
power paragraph.

### MINOR 4 -- the UNMEASURED reading prescribes "the cost shrinks with the rate" unconditionally
That sentence is true only if AO's best rate is above SO's working rate. **Fix:** "the abstract states that the cost is
unmeasured at each arm's best rate (AO@best k/8 at r_ao, SO k/8, gap d)". Add "and shrinks with the rate" only when
`r_ao > r_so_min` (MAJOR 1's definition).

### MINOR 5 -- a DIVERGED run scores accuracy 0.0, below the 0.50 predict-blank floor
NaN logits give acc 0.0 from `train_cancel.evaluate` (item 4), not "non-finite counts as 0". A diverged run at an arm's
selected rate lowers that arm's mean accuracy by ~0.12 per run, so the accuracy route there mostly measures divergence.
Divergence is data, so this is acceptable, but disclose it: "a DIVERGED run is scored at its evaluated accuracy (0.0 for NaN
weights)". Do not rescore it.

### MINOR 6 -- budget-scoping at AO's selected rate (rule 4)
If AO@best's unsolved runs are mostly DESCENDING at 300 epochs, the comparison is budget-limited, and that matters most for
UNMEASURED and SLB. The classes are printed but not used. **Fix:** the reading carries "[AO@best: x of y unsolved runs
still DESCENDING: budget-limited]" when x >= y/2.

### MINOR 7 -- `run_record` does not assert `n_trials` (400)
A hand-run or resumed checkpoint with fewer evaluation trials would be read silently. **Fix (no design content):** add
`"n_trials": 400` to `want`.

### MINOR 8 -- pilot script
- Unlike the batch driver, it has no `running` check. Restarted while its runs are in flight, it relaunches them (rule 21:
  duplicate launch, possible checkpoint overwrite).
- It records neither the code version nor a clean-tree check.
- It prints everything the extension rule needs (per-rate SOLVED and mean selection score per arm, in the secondaries
  table), and it cannot pool with the batch: the analysis defaults to `runs/lrsweep` and seeds 1500-1507.

**Fix:** copy `run_lrsweep.sh`'s `running()` guard into the loop and log `git rev-parse HEAD` plus `git status --porcelain
--untracked-files=no` at start. The pilot appears to be running already, so record in the prereg the HEAD it ran on (code
identical to 9005a48 if launched from 63aa45d) and do not restart it while runs are in flight.

### Not findings (checked; noted for the record)
- **300 epochs and low rates.** Under-training at 1e-3 / 3e-3 only weakens rates AO is not expected to select. SO is at
  ceiling there (ROBUST), so it does not affect B1 beyond MINOR 6.
- **3e-2 and the extension.** SO degrading or diverging at the top can only remove SO options, and SO is selected at its
  best. The smoke's RISING run at 2 epochs and my 100-step CPU runs are far too short to say anything about 300 epochs; they
  show only that the code runs at those rates.
- **Paired vs unpaired.** The test is unpaired, as in ROBUST, and that is consistent: the naive Fisher and permutation tests
  in the max are unpaired too. See MINOR 2 for the wording.

## Proposed amendment text (Amendment 1, before launch)

> **Amendment 1 (2026-10-16, after the independent check `docs/audits/2026-10-16/lrsweep_audit.md`; before launch, no batch
> result exists).** The check found the code doing what this file says:
> - every cell's command line sets the registered lr / wd / deltas / seeds; the 1e-3 (3e-3) line is byte-identical to ROBUST
>   def (lrx3), and the 94 shared modules are md5-identical to ROBUST's guard; bitwise on CPU at 1 thread;
> - lr reaches the single AdamW group (24 tensors, wd 0.05) at every rate including 1e-1 / 3e-1, peaks at lr after the 5%
>   warmup and decays to 0.1 x lr;
> - real NaN divergence is written, classed DIVERGED, kept and read without error;
> - the selection rule, the 12,870-split selection-aware null and the reading match independent reimplementations;
> - the driver refuses a dirty tree, a held lock and changed code, resumes without duplicates, and accepts the extension
>   edit (80 launches).
>
> Changes:
> - **Reading (audit MAJOR 1).** COST CLOSES AT HIGHER LR requires B1 = NO SIGN COST or NON-NEGATIVE LEARNS BETTER and AO's
>   best rate above `r_so_min`, the lowest rate at which SO reaches its best SOLVED count. NO COST AT BEST RATES is
>   `r_ao <= r_so_min`. "Contradicts ROBUST def / E3" is said only if the per-rate 1e-3 secondary also reads NO SIGN COST.
>   `reading()` and the selftest are updated accordingly.
> - **Decay (audit MAJOR 2).** [EITHER] A decay-matched control D is added to the batch: SO / AO at 1e-2 (wd 0.005) and 3e-2
>   (wd 1.666667e-3), lr x wd = 5e-5, seeds 1500-1507, 32 runs (~2.7 GPU-h). It is a declared secondary read per rate by the
>   ROBUST rule (Holm over 2). Under COST AT BEST RATES, a D cell reading NO SIGN COST at AO's selected rate appends "[decay-
>   coupled: at matched lr x wd the cost is not present at <rate>]". [OR] The COST AT BEST RATES scope sentence reads "with
>   AdamW weight decay 0.05 coupled to the learning rate (lr x wd 5e-5 to 1.5e-3)".
> - **Wording (MINOR 1, 2, 3, 4).**
>   - COST CLOSES / NO COST state "the cost is not detectable at AO's best tested rate; AO@best is selection-inflated" and
>     report the pooled cross-fit counts beside it.
>   - The null is "exact under exchangeability of the 16 profiles; conservative under a shared seed effect; B1's null is
>     composite and the max rule guards it (simulated SLB <= 5.2% in an adversarial composite null); with SO at ceiling the
>     decision p equals Fisher at the selected rates".
>   - Power in true rates: SLB fires 25 / 67 / 89 / 99% at AO's true best-rate solve probability 0.5 / 0.3 / 0.2 / 0.1 (SO
>     1.0).
>   - UNMEASURED's "shrinks with the rate" is said only when `r_ao > r_so_min`.
> - **Disclosed (MINOR 5, 6).** A DIVERGED run is scored at its evaluated accuracy (0.0 for NaN weights). Every reading
>   carries "[AO@best: x of y unsolved runs still DESCENDING: budget-limited]" when x >= y/2.
> - **Code, no design content (MINOR 7, 8).** `run_record` asserts `n_trials` = 400. The pilot script gets the `running()`
>   guard and logs its code version; the HEAD the pilot ran on is recorded here.

## Commands (scratchpad `lrsweep_check/`; scripts and outputs in `docs/audits/2026-10-16/lrsweep_check/`)
- selftest / power: `python3 -m mapformer.analyze_lrsweep --selftest`, `--power` (CPU), both pass / reproduce the table.
- lr: `lr_check.sh` (wraps `robust_check/wrap_gauge.py`) gives `lr_summary.txt`: groups [(0.05, lr, 24, 204243)], peak/lr
  1.000000, last/lr 0.1002 at every rate.
- bitwise: `bit3.sh` (OMP 1, seed 1581, LRSWEEP 1e-3 line vs ROBUST def line, both arms; `robust_check/cmp_ckpt.py`) gives
  "losses equal: True | tensors differing: 0", "eval.json equal: True". `bit2.sh`: identical command twice at OMP 1 is equal;
  at OMP 2 it differs (CPU threading).
- md5: `md5sum $(awk '{print $2}' runs/robust/code_md5.txt) | diff - runs/robust/code_md5.txt` shows no difference; joining the
  dry-run guard with ROBUST's gives 94 common files, 0 differences.
- divergence: `train_gauge --lr {1e2,1e3,1e4}` (6 epochs, CPU) writes NaN checkpoints and `eval.json`; those checkpoints
  put into the synthetic tree, then `analyze_lrsweep --root nantree --seeds 2000-2007 --epochs 2 --n-batches 2`, gives rc 0,
  DIVE classes, JSON written.
- independent checks: `indep.py` gives `indep_out.txt` (selection: 2 residual mismatches, both all-zero-SOLVED rounding
  ties; null n = 4: 0 / 60; n = 8 identical; 12,870 distinct rows; reading 0 / 205).
- simulations: `sim_null.py` (size under the profile null, composite nulls, power), `sim2.py` (adversarial composite nulls,
  paired variant, worst cases), `sim3.py` (winner's curse, cross-fit guard), with `*_out.txt`.
- driver: dry run plus `drvtest.sh` in a `git clone --shared` of HEAD (resume, dirty tree, lock, md5, extension).
