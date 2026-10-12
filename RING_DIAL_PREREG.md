# Howard's three cases on one task: the ring dial -- pre-registration (2026-10-12, before any run)

## Question
Claim map C7 / gap 2 (`docs/paper/CLAIM_MAP_clock_map.md`): the map-vs-clock evidence compares DIFFERENT tasks (torus vs
recency vs language) and different model sizes. Within one task family and one model, does the cost of a monotone
(Howard Case II) phase track how much the task needs a map? The ring walk (`environment_cancel.GridWorldCancel`, 32
cells, map redrawn per trajectory, runs of 1-10 repeated actions) has a dial: the probability p_plus of a +1 step.
p_plus 0.5: steps cancel, the position is a random walk (a map is needed); 0.9: one run in ten goes backwards; 1.0: every
step is +1 and the position is (start + t) mod 32, a clock (the action token is constant). Predictions (Howard 2014,
`docs/lit/LIT_CLOCK_MAP.md`): the monotone arm costs accuracy at 0.5 and 0.9 and nothing at 1.0; the index (Case I, one
layer) fails where steps cancel and solves at 1.0 (as in CANCEL: 0.717 / 0.825 / 1.000).

## Arms (`train_ring_dial.py`; VARIANT_MAP, unchanged; 1 layer, 2 heads, d 128)
| arm | case | step |
|---|---|---|
| Signed_r4 | III | `W_out W_in x` (rank 4) |
| Abs_r4 | II | `|W_out W_in x|` -- the same parameterisation with the sign removed (`model_sign.py`); the primary arm of SIGN_MATCHED |
| RoPE | I | token index |
p_plus in {0.5, 0.9, 1.0} x seeds 30-35 (n = 6; fresh for this task: CANCEL used 0-7, RANK_TESTS 20-27) = 54 runs.
Recipe = CANCEL's (T = 128, batch 128, 98 batches, lr 1e-3, cosine, 300 epochs); eval on a fresh stream (np seed 10^6,
400 trajectories) at T = 128 (primary) and 512 (OOD, secondary). Floors (CANCEL): 0.554 / 0.520 / 0.498.

## Registered decisions (`analyze_ring_dial.py`; exact permutation tests, n = 6 vs 6, minimum two-sided p 0.0022)
Primary readout: accuracy at T = 128, eval mode. A contrast FIRES if perm p < .05 and |d| >= 0.05.
- **DM (map end, p 0.5): Abs - Signed.** MONOTONE FAILS ON THE MAP: fires negative. Otherwise "NOT SHOWN" with d and CI.
- **DD (the middle, p 0.9): Abs - Signed.** MONOTONE COSTS WHERE BACKWARD STEPS ARE RARE: fires negative; otherwise NOT SHOWN.
- **DC (clock end, p 1.0): Abs - Signed.** MONOTONE MATCHES ON THE CLOCK: the 95% permutation CI lies within [-0.02, +0.02]
  (with "(both at ceiling)" when every run of both arms is >= 0.999); MONOTONE WORSE ON THE CLOCK: fires negative; else
  UNMEASURED.
- **DIAL (summary, no new test):** HOWARD CASE II PATTERN HOLDS if DM fires negative AND DC reads MATCHES; otherwise it does
  not hold as stated (each part printed).
- **DI (index check, one layer): RoPE - Signed at each p**, reported as it falls (CANCEL predicts failure at 0.5 / 0.9 and a
  match at 1.0).
Secondary: SOLVED counts (Fisher); accuracy at 512; dropout-scale re-scored accuracy (one layer; `rescore_hook`), with a FLAG
line where it would change a label; the opposition score ||D(+1) + D(-1)|| / mean ||D|| per arm.

## Process and cost
Pilot (seed 130: Signed_r4 and Abs_r4 at p_plus 0.5 and 0.9; timing and pipeline; disclosed in an amendment), independent
code audit (rule 29), batch on GPU 0 (another user holds GPU 1). ~54 runs x 5-10 min, 6 at a time: ~1-1.5 h.
