# EXPTORUS -- independent code check (rule 29), 2026-10-26

Checker: independent, results-blind (no EXPTORUS run exists; the pilot is queued, not run). Scope: `EXPTORUS_PREREG.md`,
`model_exptorus.py`, `train_exptorus.py`, `eval_exptorus.py`, `analyze_exptorus.py`, `test_exptorus.py`, `run_exptorus.sh`,
`runs/exptorus_pilot_wait.sh`, `runs/exptorus_smoke/`. Nothing launched on a GPU; no tracked file edited; nothing committed.

**Verdict: launchable after Amendment 1.** No BLOCKER. Two MAJOR: one needs a small code change (MAJOR 1), the other a
registered reading rule (MAJOR 2; coding it is optional). Five MINOR and a few NOTEs.

## What was run and what it showed

| check | result |
|---|---|
| `python3 -m mapformer.test_exptorus` (CPU) | 7/7 PASS (registration, signroute signed/abs == SignedAO4/AbsAO4 bitwise, identical init state dicts at seeds 0 and 3400, only moves step, signs, 204,757 params each, init reconstruction == lr-0 checkpoint) |
| extra: init reconstruction with `--data-workers 3` (the batch's setting; the unit test uses 0) | lr-0 CPU run, seed 3496: bitwise equal |
| `analyze_exptorus --selftest` | passes; every label, reading, batch reading and qualifier reached |
| `analyze_exptorus --power` | reproduces Sec. 5 exactly (deterministic p 0.0002 / 0.0014 / 0.0070 / 0.0256; Holm 0.0513 at k = 3 with softplus 8/8; simulated shares) |
| `analyze_exptorus --init-census --seeds 3400-3407` | reproduces the Sec. 1 table exactly |
| `pilot()` (never run by the smoke) | exercised on the smoke checkpoints with the recipe assert relaxed in a wrapper: runs; prints class, UNSTABLE, max post-warm-up loss - H_map, step max, s/epoch, projection. `epoch_seconds` parses G-4's real logs (2.4 s/epoch) |
| dry run, scratch clone (`git clone`, EXPTORUS files copied and committed IN THE CLONE ONLY), fake `nvidia-smi` on PATH, `DRV_DRYRUN=1 DRV_GPUS="5 6"` | refusals all fire: uncommitted -> REFUSED; modified tracked `stats_core.py` -> REFUSED; no DRV_GPUS -> REFUSED; pilot without ALLOW_UNCOMMITTED -> REFUSED. Committed: 64 DRYRUN launches, 64 distinct output dirs, flags identical to `run_torus_lrsweep.sh:93-95`; md5 guard over 106 files (all model modules train_variant imports, both evaluators, the analysis chain, stats_core, lib_driver, the driver); then `FAILED: missing runs` as expected |
| seeds | no `.sh`, prereg or run dir other than EXPTORUS names 34xx; RNN_TORUS uses 3300-3307 (`run_rnn_torus.sh:30`), pilots 3390/3391 |
| recipe / eval | train flags = G-4's, OMP 2; eval flags = G-4's (`run_torus_lrsweep.sh:100-105`); `check_ckpt` = G-4's plus `aux_coef` |
| step definitions | EXP / SPL come from the same `_signroute_variant` factory as SIGNROUTE's ring arms (`model_signroute.py:156-158`: `torch.exp(delta)`, `F.softplus(delta)`), = G-3's `exp(W_out W_in e)` / `softplus(...)`. train_variant has no name-dependent branch (`train_variant.py:609-614`) |
| census definitions | = `docs/audits/2026-10-25/g3_a2_exp_mech.py` (non-blind 0.05 on the first move, cancelling 0.1, k = round(S / 2 pi)); only the past-pi set differs (from the actual init, not the analytic ladder), as declared |
| waiter | detached (PPID 1, own session), so it survives this session; hard-codes `MODE=pilot` (cannot start the batch); shares `.run_exptorus.lock` with the batch (whichever starts second is refused); counts `train_rnn` as a trainer; logs its 10 h give-up |

## Findings

### MAJOR 1 -- a DIVERGED run is never UNSTABLE, so NaN divergence of exp can be read as "exp pays"
`analyze_exptorus.py:209` sets `runaway = (not div) and runaway(...)`, and `contrast()` (`:263`) counts only `runaway`
toward `unstable_at_best`. A run whose loss goes non-finite counts as unsolved with accuracy 0 and does NOT trigger the
"optimiser instability" qualifier. The other three arms are bounded or saturate. exp is the one arm that can reach float32
overflow: exp(readout) is inf once the readout passes 88.7. G-3's RUNAWAY exp runs reached steps of 3.6e24 = exp(56) in 300
epochs at 3e-3, and this batch runs 900 epochs.
**Shown** (synthetic): exp 1/8 SOLVED at 3e-3 with 5 DIVERGED runs, 0/8 at 1e-3, softplus 0/8 -> exp selects 3e-3,
`unstable_at_best 0`, reading "EXP PAYS THE SIGN COST IN ATTENTION ON THE TORUS", batch "BOTH PAY". Under the prereg's
intent (Sec. 3: an unstable cell carries no learnability sentence) this is MIXED / UNMEASURED.
**Fix**: count `cls == "DIVERGED"` as UNSTABLE for the qualifier and for `pay()` (`uns = sum((x["runaway"] or x["cls"] ==
"DIVERGED") and not x["solved"] ...)`), list DIVERGED runs in the flagged block, and add a selftest case with >= 2
DIVERGED runs at the selected rate (must read MIXED). The pilot readout should also report DIVERGED.

### MAJOR 2 -- best-vs-best inflates NO SIGN COST in favour of exp / softplus, because signed effectively has one rate
Signed at 3e-3 is known bad on the torus (G-4: 0/8, all DESCENDING), so signed's selection is fixed at 1e-3. Each
non-negative arm gets the better of two draws. For SIGNED LEARNS BETTER this asymmetry is conservative. For NO SIGN COST,
which is a descriptive rule with no p-value, it is anti-conservative, and NO SIGN COST is the reading that would carry the
headline (EXP ESCAPES).
**Measured** (simulation, 200 replicates, exp's true solve rate the same at both rates, signed 1.0 / 0.1, unsolved accuracy
N(0.88, 0.07)): P(NO SIGN COST) at a single rate vs best of two is 0.10 vs 0.18 at true rate 0.7, 0.28 vs 0.39 at 0.8, and
0.61 vs 0.77 at 0.9. So exp at a real 0.7 solve rate (a 0.3 cost against signed's 1.0) is called NO SIGN COST almost twice
as often. The cross-fit (4 seeds per half) is too weak to undo this. The prereg says "selection-inflated" but registers no
rule that uses the unselected cell.
**Fix** (reading rule; coding it is optional): the 1e-3 per-rate cell has signed at its best rate and exp / softplus at a
fixed, pre-specified rate (A2's rate), so it is unselected. A B1 NO SIGN COST (or NON-NEGATIVE LEARNS BETTER) is reported
as "at the matched rate 1e-3" only if the 1e-3 per-rate cell also reads NO SIGN COST (or better). Otherwise it reads "...
only at its selected rate <r> (selection-inflated; 1e-3 per-rate reads <L>)", and EXP ESCAPES / NEITHER PAYS / SOFTPLUS
ESCAPES inherit that wording. The rule is preferably computed in `analyse()` before the verdict JSON is written (the
per-rate labels are now computed only in `secondaries()`) and passed to `arm_reading`. Also print beside the 3e-3 per-rate
cells that they are not a sign contrast (signed is at its known-bad rate there).

### MINOR 1 -- "EXPTORUS and RNN_TORUS run sequentially" is not enforced; the waiter fires on any idle minute
The shared slot count (`lib_driver.sh:40-42` counts any `mapformer.train_` on `--device cuda:N`) makes oversubscription
impossible, and the waiter cannot launch the batch. But:
- (a) The waiter fires at the first poll with zero trainers. That will probably be the gap between RNN pilot 1 finishing
  and the RNN builder launching pilot 2 or the RNN batch, which is a human-paced gap. The RNN line is then delayed about
  40 min. If an RNN job wins a slot first, the EXPTORUS pilot's s/epoch (the only input to the 12 h rule) is measured
  beside LSTM jobs, not beside its own arms.
- (b) The waiter and `drv_pick` ignore other users' GPU processes. They check only prashr's trainers and free memory, but
  the house rule is no launches while another user's jobs run.
- (c) Batch mode does not refuse while a foreign batch runs.
**Fix**: the batch refuses if any `mapformer.train_` process outside `runs/exptorus` runs, or if `flock -n` on
`.run_rnn_torus.lock` fails. The waiter also requires no compute app from another user (`nvidia-smi
--query-compute-apps=pid` mapped to users). The pilot analysis records the trainers per GPU seen at its launches. Tell
the RNN builder that the waiter is armed.

### MINOR 2 -- applying this amendment while the pilot runs will kill the pilot's analysis
The pilot re-checks the md5 guard before its analysis (`run_exptorus.sh:223`). MAJOR 1 / 2 edit `analyze_exptorus.py`,
which is guarded. If the edit lands while the pilot trains, the pilot ends `FAILED: code changed` with no marker. The
training survives, so this is recoverable by hand. Editing `run_exptorus.sh` in place while it runs breaks bash (rule 22).
**Fix**: apply the code edits after `.exptorus_pilot_done`, or accept re-running `analyze_exptorus --pilot` by hand.
Write each file new and `mv` it over the old one.

### MINOR 3 -- the 12 h switch to option A is not wired
`RATES=(1e-3 3e-3)` is hard-coded (`run_exptorus.sh:152`). Option A (1e-3 only) needs a driver edit and `--rates 1e-3`. The
analysis has never been selftested at one rate (it runs: `select` over one column, and the "not bracketed" qualifier is
suppressed by `len(rates) > 1`).
**Fix**: state in Amendment 1 that a switch to A is a code change plus a re-check. Or pre-wire an `EXPTORUS_RATES`
override and add a one-rate selftest case.

### MINOR 4 -- confound not stated: weight decay's attractor differs by parameterisation
AdamW wd 0.05 acts on W_in / W_out (and omega). As the readout decays to 0, the step goes to 1 (exp) or ln 2 (softplus): a
uniform clock. For signed and |.| the step goes to 0. Omega also decays: RANKWD measured about 0.15 of init on this class
and recipe. The coupled decay is three times stronger at 3e-3 (exponent 7.2 vs 2.4). So under the house recipe the
optimiser pulls exp / softplus back toward the clock that must then be cancelled by wrapping. That is part of "the
parameterisation as used" and is not a flaw, but Sec. 4 and Sec. 8 should say it. A "BOTH PAY" reading must not be
worded as a property of non-negativity alone.

### MINOR 5 -- "the LRU and Mamba parameterisations" overreaches
The LRU's phase theta = exp(theta_log) is input-independent; here exp sits on a content-dependent readout. Mamba's Delta
multiplies the state matrix A. The license in Sec. 4 (BOTH PAY) should read "exp (LRU-style positivity) and softplus
(Mamba-Delta-style) content-dependent increments", not the LRU and Mamba parameterisations themselves.

### NOTEs
- UNSTABLE on exp, which starts as a clock: the loss criterion is arm-agnostic. G-4's signed runs are already at H_map by
  epoch 5 (2.0657), and a clock start cannot keep the loss above H_map + 0.05 past epoch 45 unless the run fails to learn
  the marginal, which is a real failure. The step criterion (step > 1e3) is a readout of only 6.9 for exp, against 1e3 for
  signed, so it is asymmetric. In G-3, however, solved exp runs ended at steps 4.1-7.8 and RUNAWAY ones at 1e3-3.6e24, so
  the bound separates them. The flagged block prints all three criteria. The pilot's max post-warm-up loss - H_map is the
  check; report the epoch of the maximum too.
- Holm is applied per route over EXP / SPL, but each arm's label fires on either route (OR), so the per-arm family is two
  routes uncorrected. This is the house rule (ROBUST / G-4), unchanged; say so in Amendment 1 rather than imply a
  four-test correction.
- Dead branch: `arm_reading` tests the gate before the UNREADABLE label (`:136-140`). With signed < 4/8 the gate always
  fails first, so the text blames the anchor gate ("|.| vs signed reads UNREADABLE (SO SOLVED ...)"). The "signed < 4"
  wording is unreachable, and so is "PARTLY UNREADABLE" on real data (signed's selection is shared by every contrast).
  Checking `lab.startswith(UNR)` first fixes the wording.
- The anchor gate is conservative (exp vs signed stands on its own); it is registered and accepted. If |.| happens to
  solve 4+/8 at 3e-3, the whole batch reads UNREADABLE.
- The md5 guard leaves out `mapformer/__init__.py` (43 lines, imports only) and the prereg; the clean-tree check covers
  both. Fine.
- Census: the non-blind test reads only the pair's first move, as in G-3 / G-4. A channel blind for N but not for S is
  excluded although it cannot cancel; this is descriptive only.
- Cosmetic: the Sec. 3 heading of the prereg is split over lines 96-97, so its second half renders as body text.
  `analyze_exptorus.py:128` and `:155` cite "Sec. 5" for the readings (they are in Sec. 4). `pilot()` prints "+ ~15 min
  evals" against the prereg's ~2 min. `run_exptorus.sh:113` says the smoke "exercises every command line of the batch
  except the GPU", but the smoke never runs `--pilot` (exercised here by hand: it runs).

## Proposed text

    ## Amendment 1 (independent code check, docs/audits/2026-10-26/exptorus_audit.md; before any batch result)

    1. (MAJOR 1) DIVERGED is instability. A run with non-finite training loss counts as UNSTABLE for the qualifier in
       Sec. 3 and for the batch reading: ">= 2 unsolved runs UNSTABLE or DIVERGED at the selected rate -> optimiser
       instability; no learnability sentence rests on this cell", and that arm is not read as "pays". DIVERGED runs are
       listed with the RUNAWAY ones. A selftest case (>= 2 DIVERGED at the selected rate) must read MIXED / UNMEASURED.
       The pilot readout reports DIVERGED.
    2. (MAJOR 2) Matched-rate check on any non-cost reading. Signed's selection is fixed at 1e-3 in practice (its 3e-3
       cell is known bad), so best-vs-best favours the non-negative arms for NO SIGN COST (simulated: at a true exp solve
       rate of 0.7 / 0.8 / 0.9, P(NO SIGN COST) 0.10 / 0.28 / 0.61 at one rate vs 0.18 / 0.39 / 0.77 at best of two). A
       B1 NO SIGN COST or NON-NEGATIVE LEARNS BETTER is worded "at the matched rate 1e-3" only if the 1e-3 per-rate cell
       (signed at its best, the arm at A2's pre-specified rate, no selection) also reads NO SIGN COST or better.
       Otherwise it reads "only at its selected rate r (selection-inflated; 1e-3 per-rate reads L)". EXP ESCAPES /
       NEITHER PAYS / SOFTPLUS ESCAPES inherit that wording. The 3e-3 per-rate cells are not sign contrasts (signed is
       at its known-bad rate) and are printed as such.
    3. (MINOR 1) Sequencing. The batch refuses while any mapformer.train_ process outside runs/exptorus runs or while
       .run_rnn_torus.lock is held. The pilot waiter also waits for no other user's GPU process. The pilot analysis
       records the trainers per GPU at its launches; the 12 h rule reads s/epoch only from a pilot whose four runs shared
       the GPUs with nothing else.
    4. (MINOR 2) Code edits for this amendment are applied after .exptorus_pilot_done (or the pilot analysis is re-run
       by hand), each file written new and moved into place.
    5. (MINOR 3) A switch to option A under the 12 h rule is a code change (driver RATES, analysis --rates 1e-3) and
       gets a re-check before launch.
    6. (MINOR 4) Added to Sec. 4 / 8: under the house recipe, AdamW decay pulls the exp / softplus readout toward 0,
       i.e. the step toward 1 / ln 2 (a uniform clock), and omega toward 0. For signed / |.| it pulls the step toward 0.
       Decay coupled to lr is 3x stronger at 3e-3. No reading attributes a cost to non-negativity alone.
    7. (MINOR 5) Sec. 4 BOTH PAY license: "exp (LRU-style positivity) and softplus (Mamba-Delta-style) content-dependent
       increments pay in attention on the torus", not "the LRU and Mamba parameterisations".
    8. (NOTE) Holm is per route over the two primaries; a label fires on either route (house rule, unchanged).
       arm_reading tests UNREADABLE before the gate so a signed < 4/8 batch names the right reason.
