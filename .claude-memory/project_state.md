---
name: project-state
description: Current empirical state -- what is citable, retracted, in flight and open. Cross-session handoff; verify against git log, RESULTS_INDEX.md and EM_WM_STATE.md before recommending.
metadata:
  type: project
---

## LATEST 2026-09-23 -- read this block first (full detail: CLAUDE.md's two 2026-09-23 blocks)

**Shared report.** https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc (v5), source
`report/language_summary.html`. Edit that file and republish with `url=` that link, never a new
publish. Structure the user asked for: positive results first, failures briefly at the end.

**The dividing line: matched vs mismatched length (robustness is not capability).** The two
positive results that survive are matched-length: navigation +0.461 (T=128 -> 128) and the Dyck
depth ladder (`DYCK_LADDER_RESULTS.md`: position +0.290/+0.209/+0.159/+0.168 at 1-4 layers, 8/8,
fixed width, index arms plateau ~0.76-0.78, path arms ~0.92-0.95). Every OOD-length claim that got
a matched-length control died: code C1 closed at n=3 as a REVERSAL (encoding -0.0030, MDE 0.0046,
against -3.694 extrapolating). Rank, InEKF, forget gate and PoPE-wrapping have never had one.
Practical rule: train at the target length; if you cannot, add the 48-parameter decay envelope
(RoPE + envelope was the best code arm) rather than choosing an encoding for extrapolation.

**PoPE ablation (2026-09-22, `ABLATE_RESULTS.md`)**: the non-negativity account is refuted; our
PoPE matches the authors' code to 1.7e-06; PoPE - RoPE on code/enwik8 is ~0 or worse, so their
Table 5 decomposes a gain that does not exist here.

**PoPE/MapFormer asymmetry**: PoPE's encoding helps the path row wherever it helps at all
(MapPoPE - MapWM: Bach -0.0165, Dyck 1-2L +0.073/+0.050, code -0.0052, all detectable; never
detectably worse). Adding path integration to PoPE on clock tasks hurts (code +0.0033,
detectable; Bach +0.0111, just inside MDE). Interaction (is the path-row effect larger?) NOT
established.

**bf16 NOT licensed** (`BF16_RESULTS.md`): MapWM - RoPE gap moved 0.0117 vs 0.005 threshold,
speedup only 1.25x. Keep fp32.

**In flight / next, none launched:**
- **Rank at matched length** -- audited GO, owned by the main session (`runs/rank_matched`,
  `RANK_MATCHED_PREREG.md`). r=2 vs r=4 trained and tested at T=1024, 8 seeds, `--n-steps 1024
  --batch-size 16`, else the `run_rank_sweep.sh` recipe. Audit: 94% of the old +0.085 is
  short-gap revisits late in the sequence (robustness signature); wrap revisits below floor for
  both; old training losses did not overlap. Registered expectation R1: r4 - r2 within MDE.
- **MAESTRO** (`MAESTRO_PLAN.md`): PoPE vs RoPE n=5 ~2 days, 2x2 n=3 ~2.5 d, n=5 ~4 d; miditok
  missing; effect 0.015 NLL, n=3 likely underpowered.
- **Indirect Indexing even/odd shift split** (`INDIRECT_ARITHMETIC.md`): needs a model with
  position-in-values and per-layer content-set rotation; MapWM predicted to fail by construction.
- Old navigation OOD claims (rank, InEKF, forget gate, PoPE-wrapping) need matched-length controls.

## LANDED then RETRACTED 2026-09-21 -- code modelling (the Dyck -> real-code transfer test)

> **RETRACTED by its own matched-length control** (`runs/code2048`, n=3): the OOD bpc wins below
> were the cost of extrapolating past a 512 context. At matched length the encoding effect is
> -0.0030 (MDE 0.0046) and the composition claim REVERSES. Do not cite the OOD numbers.

Full detail in CLAUDE.md's LANDED block; numbers in `CODE_RESULTS.md` / `CODE_RESULTS_OOD.md`.

- **In distribution: a CEILING.** Every arm solves bracket matching inside a 512-byte
  window (13 of 17 strata above 0.98). That is a fact about the TASK, not a model result.
- **Past the training context (4x, eval-only): MapPoPE beats BOTH components** --
  -0.102 bpc vs PoPE (MDE 0.052) and -3.804 vs MapWM, 3/3 seeds. First time the
  composition claim has fired anywhere on language-like data.
- **The ENCODING is what survives**: PoPE - RoPE = -3.585. RoPE-encoding arms fall BELOW
  the no-stack floor at long distance.
- **Corrected at n=3**: my n=1 claim that MapWM extrapolates worse than RoPE was one seed;
  at n=3 it HELPS detectably at 512-1024.
- Caveats: n=3, all runs budget-limited, and the bpc win is NOT corroborated by the
  closer-accuracy metric (unmeasured, sign against).

## LATEST (2026-09-15..20) -- Dyck-2 and the PoPE-paper line

**Nothing running, not pushed.** Full detail in CLAUDE.md's LATEST block; the citable files are
`DYCK_RESULTS_bs128.md`, `DYCK_RESULTS_POPE.md`, `DYCK_LITERATURE_METRICS.md`, `DYCK_STACK_PROBE.md`,
`INDIRECT_RESULTS_200k.md`, `JSB_RESULTS.md`, `JSB_LENGTH_RESULTS.md`, `AUG_RESULTS.md`,
`T1_RESULTS.md`, `T3_RESULTS.md`, `T2_RESULTS.md`, `DECAY_RESULTS.md`, `DYCK_DECAY_RESULTS.md`,
`CROSS_RESULTS.md`, plus `THEORY_MAPPOPE.md` (account, withdrawn as a cross-task rule).

- Dyck-2 ordering replicates, levels do not; MapPoPE-1L best (0.927). The paper's F1 has a 0.88
  no-stack floor -- use Hewitt/Yao closing accuracy and the Suzgun set criterion instead.
- Both PoPE-paper results replicate (Indirect Indexing at 200k, Bach at their recipe). Path
  integration adds nothing in distribution on either.
- Beyond the training context the naive combination COLLAPSES (4.616 vs RoPE 2.059); three repairs
  work; the account explaining them is WITHDRAWN after both halves failed within-task on Dyck.
- JSB is overfitting-limited by 3x: augmentation beats every positional effect measured on it.
- Surviving positive claim: a decay envelope's damage splits into steepness (~half) and metric
  (the rest, confounded with convergence).

New code: environment_dyck, train_dyck, analyze_dyck, validate_dyck, probe_dyck_stack,
probe_dyck_far, probe_dyck_metric, probe_dyck_alpha, eval_dyck_literature, environment_indirect,
train_indirect, analyze_indirect, eval_indirect_ood, environment_jsb, train_jsb, analyze_jsb,
analyze_jsb_len, analyze_jsb_base, analyze_t1, model_centered, model_pope_t3, model_pope_decay,
model_dyck_monotone, probe_theory_numbers, make_dyck_table, dyck_standard_metrics.

## LATEST (2026-09-12..15) -- read this block first; it supersedes "In flight" and "Open" below

**DYCK-2 (2026-09-15, `DYCK_RESULTS_bs128.md`, n=8, paper recipe, prereg `DYCK_PREREG.md`):** ordering
replicates (MapFormer-1L - RoPE-2L +0.37/+0.39 at L128 D12, 8/8), training cell replicates (0.985), OOD
levels do NOT (MapWM 0.868 / MapEM 0.888 vs paper 0.94/0.95) and sit AT a no-stack n-gram floor (0.884).
RoPE-2L matches the paper closely. Paper reports no floor; its baselines are below the n-gram. Batch size
(unstated) untested; batch-32 follow-up not triggered by the registered slope rule.
**MAPPOPE COLLAPSE (2026-09-17, partly superseded):** three repairs now work on Bach (centring,
per-token phase, decay envelope); the MECHANISM remains unidentified -- the clock/map account that
explained them was withdrawn 2026-09-20. Two pre-registered accounts refuted.
Rank: collapse survives r=1/2/4 (4.08/4.62/4.63) and the rank effect is equal on the MapWM control
row. Omega base: the predicted direction is INVERTED -- base 512 is best and 32768 worst on BOTH rows
(MapPoPE 3.82 -> 5.22, MapWM 1.11 -> 1.87), so raising the base is not the extrapolation fix here that
it is for index RoPE. MapPoPE never beats MapWM OOD at any rank or base (0/5 everywhere). Remaining
suspect (one phase per ELEMENT vs per PAIR) needs a new variant; deliberately not built.
Practical: for extrapolation on this task smaller rank and smaller base both help (MapWM r1 0.912,
MapWM base512 1.109 vs default 1.397); the combination is untested.

**MAPPOPE COLLAPSE (2026-09-17, partly superseded):** three repairs now work on Bach (centring,
per-token phase, decay envelope); the MECHANISM remains unidentified -- the clock/map account that
explained them was withdrawn 2026-09-20. Two pre-registered accounts refuted.
Rank: collapse survives r=1/2/4 (4.08/4.62/4.63) and the rank effect is equal on the MapWM control
row. Omega base: the predicted direction is INVERTED -- base 512 is best and 32768 worst on BOTH rows
(MapPoPE 3.82 -> 5.22, MapWM 1.11 -> 1.87), so raising the base is not the extrapolation fix here that
it is for index RoPE. MapPoPE never beats MapWM OOD at any rank or base (0/5 everywhere). Remaining
suspect (one phase per ELEMENT vs per PAIR) needs a new variant; deliberately not built.
Practical: for extrapolation on this task smaller rank and smaller base both help (MapWM r1 0.912,
MapWM base512 1.109 vs default 1.397); the combination is untested.

**JSB LENGTH EXTRAPOLATION (2026-09-17, `JSB_LENGTH_RESULTS.md`, n=5, train context 512):** the one
PoPE-data condition where path integration wins -- MapWM - RoPE = -0.662 NLL at 2-4x beyond the
context (5/5, MDE 0.335), growing with length. MapPoPE COLLAPSES there (4.62 vs MapWM 1.40); the
full-context control shows no collapse, so it is extrapolation-specific. PoPE's encoding is itself a
length win on the index row (-0.461).

**PoPE PAPER (2026-09-16/17):** Bach Chorales replicates (PoPE -0.032 NLL, 5/5) and path integration
adds NOTHING there (+0.011 on the PoPE row, 0/5 better) -- first clean null for path integration on a
natural-sequence task. Indirect Indexing: at the paper's 100k budget PoPE solves 1/8 and MapPoPE 5/8;
at 200k PoPE 7/8 (mean 0.965 vs paper 0.948) and MapPoPE 8/8, so the task is a late-transition search
problem, the paper replicates, and path integration buys SPEED not capability.

PoPE amendment (`DYCK_RESULTS_POPE.md`): MapPoPE-1L best arm, 0.927 at L128 D12 (+0.058 over MapWM, 8/8;
+0.312 over PoPE-1L); PoPE alone ~ RoPE; no interaction (+0.009, MDE 0.074); MapPoPE at the floor +0.042 (MDE 0.046).
PoPE amendment (`DYCK_RESULTS_POPE.md`): MapPoPE-1L best arm, 0.927 at L128 D12 (+0.058 over MapWM, 8/8;
+0.312 over PoPE-1L); PoPE alone ~ RoPE; no interaction (+0.009, MDE 0.074); MapPoPE at the floor +0.042 (MDE 0.046).

**DYCK-2 (2026-09-15, `DYCK_RESULTS_bs128.md`, n=8, paper recipe, prereg `DYCK_PREREG.md`):** ordering
replicates (MapFormer-1L - RoPE-2L +0.37/+0.39 at L128 D12, 8/8), training cell replicates (0.985), OOD
levels do NOT (MapWM 0.868 / MapEM 0.888 vs paper 0.94/0.95) and sit AT a no-stack n-gram floor (0.884).
RoPE-2L matches the paper closely. Paper reports no floor; its baselines are below the n-gram. Batch size
(unstated) untested; batch-32 follow-up not triggered by the registered slope rule.
PoPE amendment (`DYCK_RESULTS_POPE.md`): MapPoPE-1L best arm, 0.927 at L128 D12 (+0.058 over MapWM, 8/8;
+0.312 over PoPE-1L); PoPE alone ~ RoPE; no interaction (+0.009, MDE 0.074); MapPoPE at the floor +0.042 (MDE 0.046).
PoPE amendment (`DYCK_RESULTS_POPE.md`): MapPoPE-1L best arm, 0.927 at L128 D12 (+0.058 over MapWM, 8/8;
+0.312 over PoPE-1L); PoPE alone ~ RoPE; no interaction (+0.009, MDE 0.074); MapPoPE at the floor +0.042 (MDE 0.046).

**Nothing is running.** Everything below is committed locally, and **not pushed** (last commit ffa7440 or later).

**Report deliverables (`report/`).**
- `report.pdf`: 43 pages, paper-style, surviving results only, author Prashant Rangarajan.
- `report_short.pdf`: 10 pages, MDEs and seed counts stripped.
- Built from 6 inventories (`report/inventory/`), `STORY.md` ("three conditions" storyline) and two audits
  (`VERIFY.md`, `VERIFY_2X2.md`) with fixes applied.
- Every table caption is self-explanatory; the long report has a worked MDE example.
- Open cleanup: some bib entries print internal notes ("not verified") in the reference list.

**New results, one line each; the details are in the named files.**
- **MONOTONE:** Selective RoPE's generator pays the same raw sign cost on the torus (-0.355 vs MapWM -0.363).
  Monotone costs a little on recency (SRoPE -0.068 detectable, WM -0.060 unmeasured). Monotone EM pays -0.198
  but keeps a forward, wrapped rewind route. Claim is now "large on a map task, small on a counting task".
- **PAPER2X2** (converged paper-task 2x2, n=8): position +0.243 at training length, +0.359 at 8x, 8/8.
  Encoding -0.049 then +0.189. Loss-matched position is uninformative, because loss clusters do not overlap.
- **REVISIT_2X2** (eval-only): index RoPE's lead over index PoPE is at 5-16-step revisits. Beyond training length
  RoPE fails at EVERY interval (positional collapse). PoPE is robust.
- **TEM on recency** (`TEM_RECENCY_PILOT.md`, `TEM_RECENCY_DIAG.md`):
  - installed rewind 1.000;
  - from scratch at chance;
  - k=1 learned from scratch;
  - with the counter installed, rewinds are found only for k<=16.
- **COUNTER batch** (n=4): with an IDENTICAL installed counter, MapWM = 1.000 (4/4), MapEM = 0.740, TEM = 0.337.
  -> EM's deficit is NOT the counter and NOT the rank-4 bottleneck (TEM's full transform does worse). Where the
  offset is expressed (content phase vs a position-side rewind) is the surviving account. The
  "free per-block dial" explanation is dropped.
- **Continuous navigation re-read:** every model is at the copy-last-observation baseline (1.66 cells). EM's
  "flat error" means it does not break, not that it integrates better.
- **Addition line** (`ADDITION_DESIGN.md` -> `ADDITION_PILOT*.md` -> `ADDITION_CHO_REPRO.md` -> `SAMEBLOCK_*`):
  - Cho et al.'s block reproduces to 100 digits (0.938), not 200. The repo layer does not train at lr 1e-4.
  - SAMEBLOCK control gate FAILED (role-format oracle 0.697 at 100).
  - Signed MapFormer is the only learned code that learns 30-digit addition (3/3), with the predicted sign
    pattern, but it holds only to ~33-38 digits.
  - Monotone, RoPE and NoPE do not learn it.
  - The pilot's 0.79-at-60 hint (repo layer) did not replicate in Cho's block.
- **MINIWORLD_ENDPOINTS:** +0.305 (sd 0.048, MDE 0.077) and +0.015 (ceiling), both n=3.

**Infrastructure added.**
- `train_addition.py --compile --fast-data --amp`: validated at 2.3-2.9x; `environment_addition.batch_fast` is
  row-exact.
- `verify_addition_*.py`.
- `model_cho_positions.py` (swappable position mechanisms in Cho's block).
- `model_tem_recency.py`, `model_counter_installed.py`, `model_monotone.py`, `probe_revisit_2x2.py`.

**Open decisions for the user.**
1. Replicate the addition pilot's repo-layer setup at 3 seeds, to test whether the architecture made MapFormer
   generalise.
2. Dyck-2 (in paper v4, not in repo; the "Dyck infrastructure" note in `LANGUAGE_LANDSCAPE.md` is wrong).
3. Eval-only length-vs-map-size split on the PAPERTASK checkpoints.
4. Entangled-vs-factorised learning-speed / extrapolation test.
5. Push to origin.

## The research goal (stated 2026-09-08, and it reframes everything)

Not positional encoding for its own sake. **Positional encoding as the mechanism by which a
model learns a relational "where", kept separate from the "what" -- the TEM claim that
factorisation is what buys transfer to new environments and to new problems that share a
relational structure.** The axes are the design space of the where. MapEM's `A_X (*) A_P`
is TEM's `g (x) x` conjunction.

## What is citable

- **Path integration is what makes in-context cognitive maps work, and the encoding is not
  what does it.** Index arms sit ON the measured 0.506 blank floor; position +0.461,
  encoding +0.003 (unmeasured), n=8. **Every eval redraws the observation map, so these are
  transfer measurements.**
- **Use r=4, not r=2** on the MapWM family: +0.085 at T=1024 for 384 params, 8/8. It is a
  conditioning failure, not a capacity one. **Does NOT transfer to MapPoPE** (+0.019,
  unmeasured).
- **The sign of the increment is load-bearing and free** (+0.123/+0.195 over an INDEX code,
  loss-matched, 12/12). This is a replication of Sarrof / Grazzi / Selective RoPE, not ours.
- **The clock/map crossover**: forcing a monotone increment costs 0.28 on a map task and
  nothing measurable on recency. See [[project-clock-vs-map]] for its corrected scope:
  recency does not *need* a clock.
- **EM vs WM (2026-09-09..11)**: the only measured difference is recency, single-`p0` EM -
  WM = -0.375 (0/8). **It is a SEARCH problem.** The rewind solution exists (installed and
  frozen: 1.000, 8/8), can largely be held (installed trainable at 8x weight scale: 0.941),
  and is found from scratch only per query token, wrapped, for ~half the k (SEARCH_RESULTS.md;
  "0/40" withdrawn; one shared k=64 found 7/8). **Phase freedom** in `q0/k0` is real: +0.146 vs
  the matched MagOnly control (22/24; +0.113 on fresh seeds). It does not act through the
  rewind. See [[em-vs-wm-mechanism]].
- Parallel scan 2.6-3.3x vs TEM's 120x, with a mechanical reason. Loop x path integration
  composes where there is headroom. MapPoPE is the strongest single configuration on the
  paper task (`RESULTS_INDEX.md`), measured at r=2 only.

## Retracted in the EM/WM line -- do not revive

WM-is-additive (AND/OR gate); Thm 3 and its corollary; the coherence inversion; the kernel
sign as a finding (it is a gauge); N2; D4's same-seed "reproduction"; the n=8 phase /
magnitude decomposition; W4's "the landscape rejects the rewind"; the early-window
mechanism; U4's "two channels are exhaustive". One line each on why: `EM_WM_STATE.md` Sec 4.

## Live negatives -- do not re-run

Level 1.5 / InEKF is stabilisation, not inference (its benefit does not grow with the drift
it exists to correct). Refining theta across depth does nothing. PC and Kalman are duals.
MoR routing has nothing to route on. Hex does not emerge. **An explicit what/where
separator works (4.16x action-vs-observation) and buys nothing**, which corroborates
MapFormer's design rather than improving it.

## In flight (2026-09-11)

Nothing. The leakage test landed (commit 4a804e8; see the block at the end of this note).

## Open, and ranked

1. **Why only ~half the per-token rewinds are found** (SEARCH answered "is it found": yes,
   wrapped) -- k from a small set at fixed queries per token; per-epoch recording. `EM_WM_STATE.md` Sec 6 lists the rest of the EM line
   (what phase freedom does instead, which is eval-only; WM's per-pair phase spread; DOF
   torus at n=24).
2. **The OOD-length axis is unexplained.** Four mechanisms help there, alpha covers two,
   and the imported critical-dimension account is refuted.
3. **The forget-gate-as-clock test must be re-run.** It ran to completion (48/48) and the
   directory was deleted by mistake. `run_forget_clock.sh` and `FORGET_CLOCK_PREREG.md`
   are intact.
4. **MiniGrid with the full stack** (allocentric + r=4 + PoPE, never combined). It is the
   one published benchmark where this family LOSES to an index arm (0.942 vs 0.955).
5. Hierarchy at n=13-16 -- see [[project-hierarchy-negative]].
6. Transfer across a change of STRUCTURE, and sample-efficiency curves. The factorisation
   thesis needs both; neither has been measured.
7. The .tex documents (last touched 2026-09-08) do not contain the EM/WM line.

## Standing method

33 numbered rules in `CLAUDE.md`, each bought by a retraction. `RESULTS_INDEX.md` numbers
its own list differently, and its 27-28 collide with CLAUDE.md's. The rules that fire most:
gate before training; retrain every arm in one batch; put an MDE beside every contrast;
check the ceiling before pre-registering a verdict; loss-match when r(loss, acc) is large
(it has reached -0.999). From the EM line: existence before mechanism, then stability --
see [[feedback-existence-before-mechanism]].

**Kernel-theory numbers to keep (restored after consolidation):** Thm 2 |rho| effect +0.292 on the torus
(8/8, MDE 0.063) -- for a FROZEN kernel ~100x below learned amplitude, so it is not a statement about
trained EM. See `EM_WM_STATE.md` for the rest.

**Leakage test (2026-09-11, closes the EM/WM hold question):** with w_in's content columns held at zero, the
8x-installed trainable rewind = 1.000 on 8/8 (0.991 at 2x length), equal to the frozen install; leak open 0.941.
EM's recency deficit is ENTIRELY SEARCH -- exists (1.000), holdable (1.000). NOLEAK_RESULTS.md.

**SEARCH (2026-09-11, SEARCH_RESULTS.md):** found from scratch per token, wrapped (peak or trough);
fixed k=64 found 7/8 (EM faster than WM, +0.191 at 2x length, exploratory); curriculum +0.127, only
k<=32. Obstacle = spread over 64 per-token rewinds. Next: k from a small set at fixed queries per
token (**SPREAD LANDED**: at matched budget fewer offsets is better, m4-m64 +0.422 8/8; 4x
budget takes the full task 0.578 -> 0.928; the exposure-matched cells hit the ceiling so that
half is unresolved -- `SPREAD_RESULTS.md`); per-epoch per-token rewind recording. Rule 34 (CLAUDE.md): readouts must respect the model's symmetries.

**PAIRORIGIN (2026-09-12, PAIRORIGIN_RESULTS.md):** the kernel-sharing test fires. EM with per-pair
origins = 0.880 vs single-p0 0.600 vs WM 0.975 (+0.280, 7/8, MDE 0.216; within MDE of WM). Phase spread
0.000 -> 1.448, per-token-rewind route 0.962 -> 0.185. Caveats: r(loss,acc) -0.983; +2,048 params not yet
controlled (PAIRCONST queued, control has 128 MORE params). Probes must route on hasattr(m,'_origins') --
two of them mis-measured this arm, one silently.

**SPREAD2 (2026-09-12, SPREAD2_RESULTS.md):** exposure test off the ceiling (design check passes,
m4_e60 = 0.913). Matched queries-per-token: m16_e240 - m4_e60 = +0.045 (MDE 0.166) -- m stops
mattering. Fixed budget: 0.913 / 0.590 / 0.248 for m = 4/16/64, m4-m64 +0.665 (8/8). The currency is
QUERIES PER TOKEN; the number of query tokens matters only through it. Budget and steps-per-token are
not separated (exposure was varied by the budget).

**PAIRCONST (2026-09-12, PAIRCONST_RESULTS.md):** capacity control for PAIRORIGIN. Constant-origin arm
(128 MORE params) = 0.782, between P0 0.600 and EMPair 0.880. Accuracy splits +0.182 (pathway) / +0.098
(freedom), NEITHER detectable at n=8 -> attribution unresolved; needs n~36. Mechanism IS clean: the
control keeps the rewind route (0.948) and only per-pair freedom abandons it (0.189). Do not say the
kernel test 'fires' without this split.

**PAPERTASK rerun (2026-09-12, PAPERTASK_RESULTS.md):** 50 ep cosine, logs kept. Extended-length EM-WM
survives at ~2/3 size (floor-normalised +0.186 l=1024, +0.287 l=2048, both 8/8). Rule 9 r = -0.461 --
NOT a loss gap, unlike the recency line. Convergence gate (IID>=0.99) FAILED: WM is 0.968 at 50 ep vs
0.969 at 16, so the shortfall is systematic and the gate could never pass -- mis-set verdict cell.
2a stays demoted on the LENGTH-axis and cross-task counterexamples, not on floor or budget.

**PAIRSPLIT (2026-09-12, PAIRSPLIT_RESULTS.md):** at n=48 per-pair freedom DOES buy accuracy:
EMPair - EMPairConst = +0.091 (MDE 0.066, 34/48; fresh 40 seeds alone +0.089, detectable). Total is
+0.215 at n=48 = pathway +0.124 + freedom +0.091, BOTH detectable (58% / 42%). The n=8 total (+0.280)
was 30% inflated.
The n=8 batch inflated the total and the pathway term, not the freedom term -- 4th instance.

**JSB IS OVERFITTING-LIMITED (2026-09-20, `AUG_RESULTS.md`).** Pitch transposition (+/-3) gains 0.107
NLL for PoPE and 0.090 for MapPoPE (5/5 each) against the 0.032 that separates PoPE from RoPE -- so
every in-distribution encoding effect measured on this dataset sits under a ceiling 3x its size.
Ordering unchanged; the PoPE-over-MapPoPE gap WIDENS to +0.028 (detectable). Best-validation step
moves from 1000/1500 to 3000, i.e. the runs become budget-limited. Augmented PoPE reaches 0.3936
against the paper's published 0.4889.

