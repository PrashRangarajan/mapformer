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
is orientation, not a substitute. Updated 2026-09-11.

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
  and is never found from scratch (0/40). **Phase freedom** in `q0/k0` is real: +0.146 vs
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

The **leakage test** (`NOLEAK_PREREG.md`, `run_noleak.sh` -> `runs/noleak/`). Do not edit
`train_variant.py`, `train_recency.py`, `environment_recency.py` or `model_em_*.py` until
`runs/noleak/.done` exists. Check `runs/noleak_repro/DETERMINISM.txt` before reading.

## Open, and ranked

1. **Why EM never finds the rewind from scratch** -- first an init-gradient probe, then
   fixed-k or curriculum recency. `EM_WM_STATE.md` Sec 6 lists the rest of the EM line
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
