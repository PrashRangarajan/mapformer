# Does PoPE's score rescue rank 2 on the long-walk torus? -- results (2026-10-05)

Pre-registration `SCORE_RANK_PREREG.md` (+ Amendment 1 from the code audit, committed d3ebe91 before launch). Runs
`runs/score_rank/p0` (40 runs, one batch: rank-2 arms seeds 10-21, rank-4 arms 10-17, matched initialisation within
the 2x2 as stated in Amendment 1). Registered output `SCORE_RANK_ANALYSIS.txt` (`analyze_score_rank.py`, run by the
driver after md5 checks before eval and before analysis). Pilot reproduction: `Vanilla` s0 through the new wrapper
equals the stored rank_mi run on all 900 epochs (max |diff| 0.0).

## Registered: NO RESCUE

Torus, trained and tested at T=1024, 900 epochs (the RANK_MI recipe), 32 angles per head in every arm:

| arm | accuracy (held-out map) | SOLVED | classes (stalled / still descending) |
|---|---|---|---|
| MapWM r2 | 0.848 +/- 0.178 | 3/12 | 5 / 4 |
| PoPE score r2 | 0.946 +/- 0.040 | 3/12 | 2 / 7 |
| MapWM r4 | 0.9998 | 8/8 | -- |
| PoPE score r4 | 0.9995 | 8/8 | -- |
| floors on this stream | blank 0.507, best n-gram 0.576, retrace 0.843 | | |

Primary (P2 - A2): SOLVED 3/12 vs 3/12 (Fisher p 1.00); accuracy +0.098 (perm p 0.089), unmeasured below its MDE 0.155.
P2's solved count 3/12 (95% CI 0.05-0.57). The premise holds: MapWM's rank-2 deficit replicates on fresh seeds (r4 8/8
vs r2 3/12, Fisher p 0.0014), and it holds equally under PoPE's score (P4 8/8 vs P2 3/12, p 0.0014). Dropout-scale
re-score: unchanged (+0.098, p 0.091; no flag).

## Secondaries (no verdict)
- **Where PoPE's score helps, and where it does not** (revisit strata, T=1024):

| stratum | MapWM r2 | PoPE r2 | P2 - A2 |
|---|---|---|---|
| plain revisits, gap < 128 moves | 0.877 | **0.992** | +0.115 (p 0.036) |
| plain revisits, gap >= 128 | 0.749 | 0.792 | +0.043 (p 0.62) |
| **wrap-only revisits** (reachable only around the torus) | **0.647** | **0.632** | -0.015 (p 0.84) |

  PoPE's score repairs the short-gap, local part of the map; it does nothing for the wrap-only revisits, which are the
  stratum that separates rank 2 from rank 4 (both r4 arms 0.996-0.998 there). Rank 2's failure is in the periodic code
  that wrap-around demands, and no score rule changes the code.
- **Failure mode.** PoPE r2 never fails catastrophically (min 0.891 vs MapWM's 0.535): its unsolved runs are mostly
  still descending (7 vs 4), MapWM's mostly stalled (2 vs 5). Speed among runs that solve: PoPE r2 median epoch 423 vs
  MapWM 505 (n = 3 each; descriptive).
- **The clock / collapse classification (theory T1, declared secondary): "SOLVED iff a clean head" holds on 39/40 runs**
  (A2 11/12, P2 12/12, A4 8/8, P4 8/8). Rank-2 failures are CLOCK (drifting heads) in both score rules; PoPE adds two
  COLLAPSE runs. The one exception is a MapWM r2 run solved with a CLOCK head (final loss 0.050, at the threshold).
- Out of distribution (rule 10): P2 - A2 +0.109 at T=512, +0.097 at T=2048, neither p < 0.05. r(final loss, acc) -0.989.

## What it means
- **The T=128 rescue does not extend to long walks.** On the paper torus at T=128 (`MAPPOPE_PAIR_RESULTS.md`) PoPE's
  score took rank 2 from 10/16 to 16/16; at T=1024 it leaves the solved count unchanged (3/12 each). The difference is
  the wrap-only revisits: at T=128 they barely occur; at T=1024 they decide the task, and PoPE's score does not help on
  them.
- **Two separate defects, two separate fixes.** PoPE's score fixes the local, short-gap map (MapWM's content-dependent
  phase offset; `docs/theory/2026-10-05/neuro_design.md`); the rank fixes the periodic code that wrap-around needs.
  This sharpens the rank result: rank D is hard because of the exactly periodic code on the torus, consistent with
  `RANK_ND_RESULTS.md` / `RANK_WRAP_RESULTS.md` (failures concentrate on wrap-only revisits).
- The clock theory's end-state classification (T1) passes a registered-secondary out-of-sample check (39/40) on a new
  batch and a new score rule; its causal role is still untested.

## Caveats
- n = 12 / 8; one task, one length, one layer, 2 heads, 900 epochs. MapWM r2 solved 3/12 here vs 0/8 in RANK_MI (seed
  variation at this boundary); 7 of PoPE's unsolved runs were still descending, so "no rescue" is budget-scoped.
- The pilot (seeds 100-101, 40 epochs) suggested a rescue; it was a short-schedule artefact or seed luck, disclosed in
  the prereg before the batch.
- Attention/FFN initial draws differ between score rules (Amendment 1); everything else is matched.
- The strata, failure-mode and speed readouts are secondaries.
