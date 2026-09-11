---
name: MapFormer project state
description: Current empirical state, what is citable, what is retracted, and where the work sits. Cross-session handoff — verify against git log and RESULTS_INDEX.md before recommending.
metadata:
  type: project
---

Repo `/home/prashr/mapformer`, single author, pushes to `PrashRangarajan/mapformer`.
Memory mirrors to `.claude-memory/` in the repo. **`RESULTS_INDEX.md` is the
maintained current-state file; `CLAUDE.md` is the chronological log.** Read the
index first — this note is orientation, not a substitute.

## The research goal (stated 2026-09-08, and it reframes everything)

Not positional encoding for its own sake. **Positional encoding as the mechanism by
which a model learns a relational "where", kept separate from the "what" — the TEM
claim that factorisation is what buys transfer to new environments and new problems
sharing a relational structure.** The axes are the design space of the where.

That reframe is now the opening of the results paper and the intro of the review.

## What is citable (2026-09-08)

- **Path integration is what makes in-context cognitive maps work, and it is not the
  encoding.** Index arms sit ON the measured 0.506 blank floor; position +0.461,
  encoding +0.003 (unmeasured), n=8. **Every eval redraws the observation map, so
  these are transfer measurements** — a structural code reaches 0.99 across four
  tasks where an index code is at the floor.
- **Use r=4, not r=2** on the MapWM family: +0.085 at T=1024 for 384 params, 8/8.
  A conditioning failure, not capacity. **Does NOT transfer to MapPoPE** (+0.019,
  unmeasured) — PoPE already has n_blocks=64 against MapWM's 32.
- **The sign of the increment is load-bearing and free** (+0.123/+0.195 loss-matched,
  12/12) — but a replication of Sarrof / Grazzi / Selective RoPE, not ours.
- **The clock/map crossover**: forcing a monotone increment costs 0.28 on a map task
  and nothing measurable on a clock task; the same unconstrained architecture learns
  alpha 0.591 vs 0.967. The first result here showing a mechanism has a MATCH, not a
  quality. See [[project-clock-vs-map]].
- Parallel scan 2.6-3.3x vs TEM's 120x, with a mechanical reason. Loop x path
  integration composes where there is headroom. MapPoPE is the strongest single
  configuration on the paper task (0.994/0.970).

## Live negatives — do not re-run

Level 1.5 / InEKF is stabilisation not inference (its benefit does not grow with the
drift it exists to correct). Refining theta across depth does nothing. PC and Kalman
are duals. MoR routing has nothing to route on. Hex does not emerge. **An explicit
what/where separator works (4.16x action-vs-observation) and buys nothing** — which
corroborates MapFormer's design rather than improving it.

## Open, and ranked

1. **The OOD-length axis is unexplained** — four mechanisms help there, alpha covers
   two, and the imported critical-dimension account is refuted. Largest gap.
2. **The forget-gate-as-clock test** was built, chained, RAN TO COMPLETION (48/48)
   and I then deleted the directory by mistake. **Must be re-run** —
   `run_forget_clock.sh` and `FORGET_CLOCK_PREREG.md` are intact.
3. **MiniGrid with the full stack** (allocentric + r=4 + PoPE, never combined). The
   one published benchmark where this family LOSES to an index arm (0.942 vs 0.955).
   Most presentable single win available.
4. Hierarchy at n=13-16 — see [[project-hierarchy-negative]].
5. Transfer across a change of STRUCTURE, and sample-efficiency curves. Both are what
   the factorisation thesis actually needs; neither has been measured.

## Standing method

Twelve numbered rules in `CLAUDE.md`, each bought by a retraction. The ones that fire
most: gate before training, retrain every arm in one batch, MDE beside every
contrast, check the ceiling before pre-registering a verdict, and loss-match whenever
r(loss, acc) is large — it has reached -0.999.


## 2026-09-10 -- EM/WM kernel theory, tested and audited
Read AUDIT_2026-09-10.md. Citable: EM - WM = -0.375 on recency (a learnability result -- EM can
represent the task); phase freedom +0.165 (n=24, replicates on fresh seeds); |rho| +0.292 for a frozen
low-amplitude kernel. Withdrawn: Thm 3, its corollary, N2, the sign 'strengthening', the D4
'reproduction'. Open: matched-optimiser control for AlignLock; map-side DOF at n=24; forget-gate rerun.
