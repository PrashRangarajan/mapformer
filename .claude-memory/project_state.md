---
name: project-state
description: Current empirical state -- what is citable, retracted, in flight and open. Cross-session handoff; verify against git log, RESULTS_INDEX.md and EM_WM_STATE.md before recommending.
metadata:
  type: project
---

Repo `/home/prashr/mapformer`, single author, pushes to `PrashRangarajan/mapformer`.
Memory mirrors to `.claude-memory/` in the repo. **`RESULTS_INDEX.md` is the maintained
current-state file. `EM_WM_STATE.md` is the current account of the EM/WM + kernel-theory
line. `CLAUDE.md` is the chronological log, with a START HERE block at the top.** This note
is orientation, not a substitute. Updated 2026-09-15.

## LATEST (2026-09-12..15) -- read this block first; it supersedes "In flight" and "Open" below

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
