# Do the exp and softplus phase increments pay the sign learnability cost in attention on the T = 1024 torus? -- pre-registration (EXPTORUS, 2026-10-26, before any batch result)

Status: independent check done (`docs/audits/2026-10-26/exptorus_audit.md`), Amendment 1 applied (end of file); NOT launched. Not yet committed. The pilot (Sec. 9) ran on outside
seeds before this file was frozen; its outcome is disclosed there.

## 0. Question

After G-3 block A2 (`G3_SSM_A2_RESULTS.md`, registered) the sign claim is "the |.| and softplus increments pay; exp does
not in the SSM at lr 1e-3" (exp 6/8 vs signed 7/8, NO SIGN COST; it cancels by wrapping, 90 of its 95 cancelling channels
among those its initialisation put past pi). The critical review (`docs/audits/2026-10-26/CRITICAL_REVIEW.md`, Part B
BLOCKER B-B2) notes that exp -- the LRU's parameterisation -- has never been run in attention on the headline task (the
T = 1024 torus): not in SIGN_MATCHED, GENERALITY, TORUS_START or TORUS_LRSWEEP. Softplus (Mamba's Delta) was run only with
observation steps (SIGN_MATCHED Pos, -0.218, 1/8). Without this batch the paper cannot say whether its sign sentence covers
the parameterisations people use, nor whether exp's escape in the SSM is a property of exp or of the SSM / ring.

**Primary question.** On the T = 1024 torus, in one-layer softmax attention, with each arm at its own best of two learning
rates: does the exp increment learn the map as well as the signed one (as in the SSM), and does softplus?

## 1. Design

**Task and recipe** = TORUS_LRSWEEP (G-4) / GENERALITY E3, unchanged: `train_variant` (entered through `train_exptorus`, which
only registers two class names first), 64 x 64 torus, 16 object types, p_empty 0.5, T = 1024 moves (2048 tokens), batch 16,
98 batches per epoch, 900 epochs, cosine (5% linear warm-up, cosine to 0.1 x lr), AdamW wd 0.05 (coupled: lr x wd per
step), 1 layer, 2 heads, d 128, `--fast-attn`, `--data-workers 3`, OMP 2, gradient clip 1.0. Observation-step-free rank-4
class: only the four move tokens (N, S, W, E = ids 0-3) step; every other token's step is exactly 0.

**Arms** (the readout function is the only difference; `test_exptorus.py` measures each claim):

| arm | class | move step (before omega) | init step mean (seeds 0-3) |
|---|---|---|---|
| SO signed (anchor) | `SignedAO4_r4` (`model_wrap_exist`) | W_out W_in e(m) | -0.013, 50.5% negative |
| AO \|.\| (anchor gate) | `AbsAO4_r4` (`model_wrap_exist`) | \|W_out W_in e(m)\| | +0.251 |
| EXP (primary) | `ExpAO4_r4` (`model_exptorus`) | exp(W_out W_in e(m)) | +1.045, all > 0 |
| SPL (primary) | `SoftplusAO4_r4` (`model_exptorus`) | softplus(W_out W_in e(m)) | +0.701, all > 0 |

EXP and SPL are `model_signroute`'s `exp` / `softplus` modes (SIGNROUTE's ring arms) with the torus's four moves. Measured:
the signroute `signed` / `abs` modes with moves 0-3 are bitwise SignedAO4_r4 / AbsAO4_r4 (state dict, step table,
forward); at one seed all four arms' state dicts are bitwise identical; parameter count 204,757 in every arm; only the
moves step; exp / softplus steps strictly positive. So **at a seed every arm and rate starts from the same weights and sees
the same training stream**; the arms differ only by the function on the readout, the rates only by lr. "Where the
parameterisation allows": the WEIGHTS are shared, the initial STEPS are not -- exp(0) = 1 and softplus(0) = ln 2 mean the
two non-negative arms start as a RoPE-like clock with large positive steps (SIGNROUTE's init confound). That start is the
hypothesised mechanism, so it is recorded (Sec. 6), not removed.

**Initial angle sums vs pi** (computed from the initial weights only -- no training -- `analyze_exptorus --init-census
--seeds 3400-3407`; per pair N+S / W+E, raw pair sum omega_c (a_c(x) + a_c(y)) over the 64 channels (2 heads x 32); "past
pi" = |sum| > pi, from where the nearest cancelling point is the 2 pi wrap, not zero):

| arm | past pi N/S, per seed 3400-3407 | past pi W/E | past pi on both pairs | median \|sum\| (rad) |
|---|---|---|---|---|
| signed | 2 2 3 1 1 3 5 4 | 5 4 1 4 0 2 3 3 | 1 2 1 1 0 1 2 0 | 0.23 |
| \|.\| | 2 2 3 1 1 5 5 4 | 5 4 3 10 2 2 5 4 | 1 2 1 1 0 2 3 1 | 0.36 |
| exp | 23 22 24 23 22 23 21 23 | 20 20 23 24 23 21 21 24 | 18 20 21 23 21 18 19 22 | 1.58 |
| softplus | 16 13 15 16 14 15 16 16 | 13 13 15 16 16 16 17 18 | 12 12 14 15 13 13 16 16 | 1.06 |

The grid-64 omega ladder (omega_i = 2 pi (1/64)^(i/31)) gives the analytic counts 11 (exp, step 1: 2 omega > pi) and 8
(softplus, step ln 2) channels per head, 22 / 16 per model -- the measured counts match. G-3's SSM ladder (grid 32) gave 13 of
32 per head for exp and 9-10 for softplus: the torus start puts a SMALLER share of channels past pi (34% vs 41% for exp). Per seed, the both-pairs count is the
relevant one for the torus (a channel must cancel N+S and W+E): exp 18-23, softplus 12-16, signed / |.| 0-3 of 64.

**Learning rates: each arm at its own best of {1e-3, 3e-3}** (decision below). 4 arms x 2 rates x 8 seeds = **64 runs**.

**Seeds 3400-3407**, fresh: no run directory under `runs/` matches `*_s34[0-9][0-9]*` and no driver or prereg names them
(checked 2026-10-26). 3300-3307 were first chosen and dropped because the concurrent RNN_TORUS build (`run_rnn_torus.sh`)
uses them, and its pilot uses 3390. The pilot here uses 3490-3491, the CPU smoke 3498-3499, the unit test's lr-0 run 3497
(all outside the batch; never pooled). Training streams (batch seed
s x 1000003 + i, i < 88,200) cannot overlap between seeds.

**Evaluation** (after training; G-4's): `eval_noise_refine` at T = 1024 (primary accuracy) and 2048, 100 walks, held-out map
(env seed 10000, walk seed 1234 + s); `eval_rank_strata` at T = 1024 (revisit strata), both entered through `eval_exptorus`
(registration only). Floors (T = 1024 eval stream, `docs/audits/2026-10-20/WHAT_WHERE_T1024.md`): blank 0.506, best n-gram
0.575, retrace 0.843.

## 2. The rate decision and its cost (measured, not guessed)

Measured on this host and recipe (G-4 log, `torus_lrsweep.log`, kalman GPU 1, 2 jobs per GPU): one wave of 2 runs = 35.5-
36 min (2.4 s/epoch x 900); evaluation of 48 runs ~2 min. The batch runs 2 jobs per GPU on both kalman GPUs = 4 slots.

| option | runs | waves of 4 | training | + evals | fits ~10 h? |
|---|---|---|---|---|---|
| A: one rate, 1e-3 (A2's rate; signed's torus best) | 4 x 8 = 32 | 8 | 8 x 36 min = 4.8 h | ~5.0 h | yes, 5 h spare |
| B: each arm at its best of {1e-3, 3e-3} (G-4's B1) | 4 x 2 x 8 = 64 | 16 | 16 x 36 min = 9.6 h | ~9.8 h | yes, at the edge |
| (B': non-negatives at both rates, anchors at 1e-3; not offered -- no symmetric selection-aware test) | 48 | 12 | 7.2 h | ~7.4 h | -- |

**Chosen: B.** Reasons: (1) a parameterisation change is an optimiser change (rule 15), and the sign line's current standard
on the torus is best-vs-best (G-4: COST AT BEST RATES); a one-rate exp / softplus contrast reopens the objection G-4 closed.
(2) |.|'s torus best was 3e-3, not 1e-3 (G-4), and nothing is known about exp's or softplus's torus optimum; the SSM
showed exp's outcome flips with the rate (3e-3 8/8 RUNAWAY, 1e-3 6/8). (3) B fits the ~10 h budget on two GPUs. A2's
exact rate is kept as a declared per-rate secondary (the 1e-3 cell). The pilot's 4-concurrent timing (Sec. 9) is the check
on (3): **if it projects the batch above 12 h, an amendment switches to A before launch.** Nothing else in the pilot can
change the design.

Expected wall time with B: ~9.8 h, both kalman GPUs fully used (no other job may share them: the LSTM/GRU torus batch must
run before or after, not beside -- lib_driver's slot count is shared across drivers, so a concurrent batch would stretch both).

## 3. Registered decisions (`analyze_exptorus.py`; machinery imported unchanged from `analyze_lrsweep.py` / `analyze_torus_lrsweep.py`)

SOLVED = `stats_core.classify_run` (final-5% mean training loss < 0.05). DIVERGED (non-finite loss) = not SOLVED, accuracy 0.
Accuracy = held-out accuracy at T = 1024 (non-finite counts 0). **UNSTABLE** (G-4's rule, unchanged): RUNAWAY = an epoch
after the 5% warm-up with training loss above H_map(seed) + 0.05 (the run's own context-free loss), or a final move step
above 1e3; COLLAPSED = final-5% loss above 1.9.

**B1 per arm, best vs best** (G-4 / LRSWEEP rule, unchanged). Selection (training-side only): an arm's best rate = most
SOLVED of 8, ties -> lower mean selection score (final-5% loss capped at the first-epoch loss), then the lower rate. The
signed arm's selected rate is the same in every contrast (it depends only on the signed profiles). Test: the
selection-aware permutation null over all C(16, 8) = 12,870 splits of the 16 per-seed rate profiles (8 signed + 8 of the
arm), the selection redone in each split; decision p per route = max(selection-aware p, naive p at the selected rates)
(Fisher on SOLVED, unpaired permutation on accuracy).

**Primaries: EXP vs SO and SPL vs SO, Holm over the two** on each route's decision p (step-down; the ROBUST rule then reads
the Holm-adjusted p). Labels (`label_from` = the ROBUST cell rule), in order:
- **UNREADABLE** if signed@best solves < 4/8;
- **SIGNED LEARNS BETTER** if p_SOLVED(Holm) < .05 with signed ahead, or p_acc(Holm) < .05 with d = arm - signed <= -0.05;
- **NON-NEGATIVE LEARNS BETTER**, the mirror;
- **NO SIGN COST** if signed@best >= 6/8, arm@best >= signed - 1 SOLVED and mean accuracy >= signed - 0.01;
- else **UNMEASURED**.

**Anchor gate: AO vs SO must read SIGNED LEARNS BETTER** (same B1 machinery, raw decision p; it is a replication of G-4's
primary on fresh seeds, not a test). If it does not, both primaries read **UNREADABLE (anchor gate)**: a batch that does not
reproduce the torus sign cost cannot say whether exp / softplus pay it.

**Qualifiers** (appended to the reading, G-4's): an arm's best rate is the grid's top rate (3e-3) -> "optimum not
bracketed; 'best' = best TESTED"; DESCENDING runs at the selected rate -> "budget-scoped"; >= 2 unsolved runs UNSTABLE at
the selected rate -> "optimiser instability; no learnability sentence rests on this cell" (and that arm is not read as
"pays" in the batch reading).

## 4. Readings (registered map; `arm_reading`, `batch_reading`)

Per primary arm X in {exp, softplus}:
- SIGNED LEARNS BETTER -> "X PAYS THE SIGN COST IN ATTENTION ON THE TORUS";
- NO SIGN COST -> "NO SIGN COST FOR X IN ATTENTION ON THE TORUS (at its best tested rate; selection-inflated, see the
  cross-fit and the per-rate cells)";
- NON-NEGATIVE LEARNS BETTER -> "X LEARNS BETTER THAN SIGNED ON THE TORUS";
- UNMEASURED -> "UNMEASURED"; UNREADABLE as above.

Batch reading (one line for the paper's sign sentence): **BOTH PAY** (exp's SSM no-cost does not transfer to attention on
the torus; the sign sentence may say that the exp (LRU-style positivity) and softplus (Mamba-Delta-style) content-dependent
increments pay in attention -- not 'the LRU and Mamba parameterisations' themselves [wording per Amendment 1.7]); **EXP ESCAPES, SOFTPLUS
PAYS** (exp's no-cost replicates in attention on the torus; the claim stays "|.| and softplus"); **NEITHER PAYS** (the cost
is specific to |.| among the tested increments); **SOFTPLUS ESCAPES, EXP PAYS** (not predicted); **MIXED / UNMEASURED**;
**UNREADABLE / PARTLY UNREADABLE**.

**What each reading licenses, and what it does not.**
- BOTH PAY: in one-layer attention on the T = 1024 torus, at the best of two rates, the exp and softplus increments learn
  the map worse than the signed one. It does NOT say why exp escaped in the SSM: the SSM batch was a ring (32 cells, T 128,
  300 epochs) and this is a torus, so mixer and task are confounded across the two batches (never pooled). SIGNROUTE (ring,
  attention: exp 1/8 at both rates) is the one cross-batch fact pointing at the mixer.
- EXP ESCAPES: exp's no-cost holds in attention on the torus. It does NOT separate exp's past-pi start from its
  multiplicative form (d step / d s = step): both are properties of the parameterisation here, as in A2. The declared
  census (Sec. 6) can show whether the cancelling channels are the past-pi-start ones; it cannot show causation. A follow-up
  that moves exp's start below pi (shifting the pre-exp readout) would separate them; it is not in this batch.
- NO SIGN COST is read at the arm's selected rate and is selection-inflated in its favour; the cross-fit and the per-rate
  cells are reported beside it.

## 5. Power (`analyze_exptorus --power`, synthetic; signed (8/8, 0/8) by rate as in G-4)

Deterministic profiles (an arm solving k/8 at its best rate, unsolved accuracy 0.90): with softplus at 0/8, exp reads
SIGNED LEARNS BETTER at k <= 3 (decision p 0.0002 / 0.0014 / 0.0070 / 0.0256), UNMEASURED at k = 4-7, NO SIGN COST at 8/8
(and at 7/8 only if the unsolved run's accuracy keeps the mean within 0.01 of signed's). With softplus at 8/8 (so Holm
doubles exp's p), k = 3 falls to UNMEASURED (Holm 0.051). Simulated (400 replicates, binomial seeds, unsolved accuracy
N(0.88, 0.07)): exp true solve rate 0 -> SLB 1.00; 0.2 -> SLB 0.86; 0.4 -> SLB 0.33 / UNM 0.66; 0.6 -> UNM 0.88; 0.8 -> NSC
0.49 / UNM 0.51; 1.0 -> NSC 1.00. So: a cost as large as |.|'s is detected with certainty; NO SIGN COST needs exp to solve
on ~7-8 of 8 seeds at its best rate; a 40-60% solve rate is UNMEASURED.

## 6. Declared secondaries (no verdict depends on them)

1. **Per-rate contrasts** (each arm vs signed at the same rate; Holm over the two rates within each arm and route; ROBUST rule
   on Holm p). The 1e-3 cell is A2's rate (the matched-rate replication question).
2. **Mechanism census** (per run, from the checkpoint and the reconstructed initial model; `cancel_census`): per (pair,
   channel) cell, non-blind (|wrap(first-move angle)| > 0.05), cancelling (|wrap(pair sum)| < 0.1), **by a wrap**
   (round(sum / 2 pi) != 0) or **through zero** (= 0); overlap with the initial past-pi set: cancelling channels that started
   past pi, P(cancel | past-pi start) vs P(cancel | not); turns; omega end / init. Reported per arm and rate, and for solved
   vs unsolved runs. This is G-3's `g3_a2_exp_mech.py` readout with the initial set measured from the actual initial weights
   instead of the analytic step-1 ladder.
3. **Initial past-pi record** (Sec. 1 table; re-printed by the analysis from each run's reconstructed init).
4. **RUNAWAY / COLLAPSED flags**, every run listed with max post-warm-up loss - H_map, final loss, step max.
5. **Revisit strata** at T = 1024 (plain lag < 128 | >= 128 | wrap-only) per arm and rate.
6. Threshold sensitivity (SOLVED at loss 0.1, selection redone); accuracy at T = 2048 at the selected rates (robustness, not
   capability); cross-fit (rates chosen on seeds 3400-3403, tested on 3404-3407, and the reverse); usage-weighted cancelling
   share and opposite-sign share (as G-4); rule 2 r(log10 final loss, accuracy); the decay exponent per rate (2.4 / 7.2).

## 7. Predictions (before any batch result; the pilot in Sec. 9 is exp-only and disclosed)

| quantity | prediction |
|---|---|
| anchor gate (\|.\| vs signed, best vs best) | SIGNED LEARNS BETTER, ~0.95 (G-4: 0/8 vs 8/8; GENERALITY E3 the same) |
| signed selected rate | 1e-3 (G-4: 8/8 at 1e-3, 0/8 DESCENDING at 3e-3), ~0.9 |
| \|.\| | 0-1/8 at both rates |
| exp | SIGNED LEARNS BETTER ~0.6, UNMEASURED ~0.25, NO SIGN COST ~0.15 (attention ring 1/8 at both rates; the past-pi start did not rescue \|.\| on the torus (TORUS_START 0/8); but A2 escaped) |
| softplus | SIGNED LEARNS BETTER ~0.75, UNMEASURED ~0.2, NO SIGN COST ~0.05 |
| exp UNSTABLE at 3e-3 (>= 2 runs) | ~0.3 (SSM 8/8 RUNAWAY at 3e-3; attention ring none) |
| batch reading | BOTH PAY ~0.5, MIXED / UNMEASURED ~0.3, EXP ESCAPES ~0.15, other ~0.05 |
| census, if exp solves any run | its cancelling channels lie mostly in the past-pi start set and cancel by a wrap (as in A2) |

## 8. Scope and what this cannot show

One layer, one task (T = 1024 torus, observation-step-free rank 4), two rates, 900 epochs, wd 0.05 coupled to lr. It does not
vary the start independently of the parameterisation (see Sec. 4), the mixer within one task (SSM on the torus is not run),
or the budget. A NO SIGN COST is "at its best tested rate". 3e-3 is signed's known bad rate on the torus (all DESCENDING in
G-4), so signed is effectively at 1e-3 in every contrast.

## 9. Pilot (outside seeds 3490-3491; disclosed; not pooled)

Purpose: timing at the batch's concurrency (2 jobs per GPU, both GPUs) and exp's stability at both batch rates. Runs: exp
at 1e-3 and 3e-3, seeds 3490-3491, 900 epochs, the batch's command line (`MODE=pilot`); analysis `analyze_exptorus --pilot`
(class, UNSTABLE flag, final loss, step max, s/epoch, projected batch time). The pilot's solve outcomes are visible to the
builder; they do not enter the design except through the 12 h timing rule in Sec. 2.

Pilot status (2026-10-26 01:51): QUEUED, not yet run. Both kalman GPUs were taken by the RNN_TORUS lr pilot (14 jobs,
2 per GPU, until ~04:00-04:30), so the pilot could not run at the batch's concurrency. A detached waiter
(`runs/exptorus_pilot_wait.sh`, log `exptorus_pilot.log`) starts `MODE=pilot` as soon as no `mapformer.train_` process
runs (gives up after 10 h); its result lands in `runs/exptorus_pilot/PILOT_ANALYSIS.txt` (marker `.exptorus_pilot_done`)
and goes into Amendment 1 before launch. Until then the timing is G-4's measured 2.4 s/epoch at 2 jobs per GPU (same
model size, recipe and host); exp's step is the same compute as signed's. Readout the amendment must report: per run
class, UNSTABLE flag, max post-warm-up loss - H_map, step max, s/epoch, projected batch hours (12 h switch rule, Sec. 2).

CPU smoke (done, 2026-10-26 01:48): `MODE=smoke ALLOW_UNCOMMITTED=1 bash run_exptorus.sh`, 16 tiny runs (all 4 arms x 2
rates x seeds 3498-3499) through train_exptorus, both evaluators through eval_exptorus, and `analyze_exptorus --smoke`,
end to end, no warning (`runs/exptorus_smoke/SMOKE_ANALYSIS.txt`). The loader and census were also run read-only on G-4's
real 900-epoch SignedAO4_r4 / AbsAO4_r4 checkpoints (seeds 2400-2401, both rates): recipe asserts pass, classes and
accuracies equal G-4's per-seed table, signed solved runs cancel 100% through zero, |.| cancels on 5-13 of 25-49 non-blind
cells, mostly by a wrap.

## 10. Files and launch

New files only (no existing module changed): `model_exptorus.py` (two classes + `register()` into
`train_variant.VARIANT_MAP` at runtime, refusing to shadow a different class), `train_exptorus.py` (registers, then
`train_variant.main()`), `eval_exptorus.py` (registers, then `eval_noise_refine` / `eval_rank_strata`), `analyze_exptorus.py`
(`--selftest`, `--power`, `--init-census`, `--pilot`, `--smoke`), `test_exptorus.py` (7 unit checks incl. init
reconstruction against an lr-0 checkpoint), `run_exptorus.sh` (modes batch / pilot / smoke; sources `lib_driver.sh`: flock
`.run_exptorus.lock`, clean tree for tracked *.py / *.sh and committed EXPTORUS files + dependencies in batch mode, md5 guard
over 106 files re-checked before every launch, DRV_GPUS / DRV_MAXPG required, kalman only, artifacts-then-marker
`.exptorus_done`).

Launch (after the independent check's amendment and a commit of the EXPTORUS files):

    cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_exptorus.sh > /dev/null 2>&1 &

Outputs: `runs/exptorus/<tag>/<arm>/p0/<V>_s<S>/`, `runs/exptorus/eval/`, `EXPTORUS_ANALYSIS.txt`, `EXPTORUS_VERDICTS.json`,
log `exptorus.log`, marker `.exptorus_done`.

## Amendment 1 (independent code check, `docs/audits/2026-10-26/exptorus_audit.md`; 2026-10-26, before any pilot or batch result)

The check found no BLOCKER, 2 MAJOR, 5 MINOR. Every item below is applied in code (new files only; no existing module
changed) and covered by the selftest where it is a rule. No GPU run of this design has produced a result yet: the waiter
that would have started the pilot was stopped at 02:05 before it launched anything.

1. **(MAJOR 1) DIVERGED is instability.** A run with a non-finite training loss now counts toward the UNSTABLE qualifier
   (Sec. 3): ">= 2 unsolved runs UNSTABLE (RUNAWAY / COLLAPSED / DIVERGED) at the selected rate -> optimiser instability;
   no learnability sentence rests on this cell". Such an arm is read neither as "pays" nor as "escapes" in the batch
   reading. DIVERGED runs are listed with the RUNAWAY ones, with the epoch of the maximum post-warm-up loss. The pilot
   readout marks DIVERGED runs as UNSTABLE. Selftest: exp 1/8 SOLVED at 3e-3 with 5 DIVERGED runs there, 0/8 at 1e-3,
   softplus 0/8 -> exp selects 3e-3, `unstable_at_best` 5, batch reading MIXED / UNMEASURED (before this fix it read a
   clean "EXP PAYS" and "BOTH PAY").
2. **(MAJOR 2) Matched-rate check on any non-cost reading.** Signed's 3e-3 cell is known bad on the torus, so its selection
   is fixed at 1e-3 in practice, and best-vs-best favours the non-negative arms for NO SIGN COST. The audit simulated this:
   at a true exp solve rate of 0.7 / 0.8 / 0.9, P(NO SIGN COST) is 0.10 / 0.28 / 0.61 at one rate vs 0.18 / 0.39 / 0.77 at
   best of two. Rule: a B1 NO SIGN COST or NON-NEGATIVE LEARNS BETTER is worded **"at the matched rate 1e-3"** only if the
   1e-3 per-rate cell also reads NO SIGN COST or NON-NEGATIVE LEARNS BETTER. That cell has signed at its best rate and the
   arm at A2's pre-specified rate, with no selection; it uses the per-rate Holm over the rates, ROBUST rule. Otherwise the
   reading is **"... only at its selected rate r (selection-inflated; 1e-3 per-rate reads L)"**. EXP ESCAPES / NEITHER
   PAYS / SOFTPLUS ESCAPES carry that qualifier into the batch reading. The per-rate labels are computed in `analyse()`
   before the verdict JSON is written (key `per_rate`; per arm `label_matched_1e3`). The 3e-3 per-rate cells are printed
   as "not a sign contrast" (signed is at its known-bad rate there). Selftest: exp 8/8 at 1e-3 -> "at the matched rate
   1e-3"; exp 2/8 at 1e-3 and 8/8 at 3e-3 -> "only at its selected rate 0.003 (selection-inflated; 1e-3 per-rate reads
   SIGNED LEARNS BETTER)", and the batch reading carries it.
3. **(MINOR 1) Sequencing.** The pilot waiter is retired (`runs/exptorus_pilot_wait.sh` deleted; it had been stopped at
   02:05 before launching anything; see `exptorus_pilot.log`). Pilot and batch are launched by hand, after a manual
   `nvidia-smi` check for other users' processes. The driver also REFUSES to start (pilot and batch) in any of these cases:
   - a `mapformer.train_` process runs outside the mode's run dir;
   - `.run_rnn_torus.lock` or `.run_rnn_torus_pilot.lock` is held;
   - another user owns a GPU compute process.

   Each launch appends `launch <rate> <arm> s<seed> cuda:<g> <time> g0=.. g1=.. foreign_trainers=.. other_users=..` to
   `<run dir>/launch_occupancy.txt`. The pilot analysis prints that file and states whether the timing is readable for
   the 12 h rule: only when no launch shared the GPUs with a job outside the pilot.
4. **(MINOR 2)** Applied before the pilot (no pilot was running), each edited file written new and moved into place.
5. **(MINOR 3) The 12 h switch is a declared line, not a code change.** The batch reads its rates from exactly one line in
   this file, beginning `EXPTORUS_RATES = `. It refuses unless that line is "1e-3 3e-3" (option B) or "1e-3" (option A),
   and passes the rates to the analysis. A switch to option A under the 12 h rule is a further amendment that edits only
   this line. It needs no re-check: the one-rate analysis is selftested (exp 8/8 -> NO SIGN COST at the matched rate,
   softplus 1/8 -> SIGNED LEARNS BETTER; no "not bracketed" or selection clause). The pilot always runs both rates. The
   declared line:

EXPTORUS_RATES = 1e-3 3e-3

6. **(MINOR 4) Weight-decay confound (Secs. 4 and 8).** Under the house recipe AdamW decay (wd 0.05, coupled to lr) acts on
   W_in / W_out and omega. As the readout decays toward 0:
   - exp / softplus steps go to 1 / ln 2: a uniform clock that must then be cancelled by wrapping;
   - signed / |.| steps go to 0;
   - omega decays too (about 0.15 of init on this class and recipe, RANKWD; 0.09-0.20 on G-4's seeds 2400-2401 at 1e-3, 0.005-0.10
     at 3e-3).

   The coupled decay is 3x stronger at 3e-3 (exponent 7.2 vs 2.4). This attractor is part of "the parameterisation as
   used under the house recipe" and is not removed. No reading attributes a cost to non-negativity alone: BOTH PAY means
   these two increments under this recipe.
7. **(MINOR 5) Wording.** The BOTH PAY licence (Sec. 4) and the batch reading now say: "the exp (LRU-style positivity) and
   softplus (Mamba-Delta-style) content-dependent increments". They no longer say "the LRU and Mamba parameterisations".
   The LRU's phase is input-independent, and Mamba's Delta multiplies the state matrix.
8. **(NOTEs)**
   - Holm is per route over the two primaries, and a label fires on either route: the house rule (ROBUST / G-4),
     unchanged, not a four-test correction.
   - `arm_reading` now tests UNREADABLE (signed < 4/8) before the anchor gate, so that case names the right reason
     (selftest).
   - The step-RUNAWAY bound (1e3) is asymmetric: it is a readout of about 6.9 for exp. G-3 separated solved exp (steps
     4-8) from RUNAWAY exp (1e3-3.6e24) with it, so it is kept. The loss criterion is reported with the epoch of its
     maximum.
   - Cosmetics fixed: the Sec. 3 heading on one line; Sec. references in the code; the pilot's eval estimate (~2 min);
     the smoke's description in the driver (it does not exercise `--pilot`, which the checker ran by hand); the
     analysis header prints the evaluated length.

**Re-verification after the amendment** (CPU, 2026-10-26): `test_exptorus` 7/7; `analyze_exptorus --selftest` passes
(every label, reading, batch reading, the DIVERGED, matched-rate and one-rate cases); CPU smoke re-run end to end;
dry run in a scratch clone (Sec. "Launch" below).

**Launch procedure (by hand, sequential, kalman):**
1. Pilot, after the RNN lr pilot has ended. Check first: `nvidia-smi` shows no process (or only prashr's, none of them a
   trainer), and `ls -la .run_rnn_torus*.lock` holds no live lock.
   `cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 MODE=pilot ALLOW_UNCOMMITTED=1 setsid nohup bash run_exptorus.sh > /dev/null 2>&1 &`
   (drop ALLOW_UNCOMMITTED once the EXPTORUS files are committed). Its result goes into Amendment 2 with the 12 h rule
   applied: projection = 16 waves x 900 x s/epoch.
2. Batch, after Amendment 2, a commit, and the same manual check:
   `cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_exptorus.sh > /dev/null 2>&1 &`
