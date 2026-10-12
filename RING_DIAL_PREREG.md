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

## Amendment 1 (2026-10-12, after the independent code audit `docs/audits/2026-10-12/ring_dial_audit.md`; before the batch)
**Pilot (seed 130, disclosed; read before the audit was written into this amendment).** Signed_r4 1.000 / 1.000 at p_plus
0.5 / 0.9; Abs_r4 1.000 / 0.979 (T = 128); at T = 512 Signed 0.999 / 0.981, Abs 0.928 / 0.841. A post hoc probe found no
clean per-channel wrap in the pilot Abs model (median |wrap(step(+1) + step(-1))| 0.79 rad vs Signed 0.02-0.03).
**M1 (decisive): Abs is not monotone for a rotary phase.** Angles wrap mod 2 pi: Delta(-1) = 2 pi / omega - Delta(+1) gives the
signed code exactly with non-negative steps (constructed by the audit in the real modules, 6e-4 over 512 steps). Any Abs
deficit is therefore a SEARCH / learnability failure, and Abs is not Howard's Case II. The original prediction ("monotone
fails on the map") is withdrawn. **The question becomes a LEARNABILITY dial: does the non-negative parameterisation find the
map as reliably as the signed one, and does that depend on how much the task needs cancellation?** (Records stating that a
monotone rotary step "cannot represent a -1 action" are corrected: CLAUDE.md sign row, memory, claim map, lit note.)
**M2.** p_plus 1.0 is true by construction (constant action token) and is dropped; the dial is **p_plus 0.5 / 0.75 / 0.9**.
**M3.** **n = 8** (seeds 30-37) and SOLVED (Fisher) is co-primary with accuracy.
**Arms.** Signed_r4 and Abs_r4 (RoPE dropped: CANCEL measured one-layer index at these p_plus with n = 8). 2 x 3 x 8 = 48 runs.
**Decisions (replace DM / DD / DC / DIAL / DI), per p_plus, d = Abs - Signed:** ABS LEARNS WORSE: accuracy fires negative
(perm p < .05 and |d| >= 0.05) OR Fisher p < .05 with fewer Abs SOLVED. ABS LEARNS BETTER: the mirror. ABS LEARNS AS WELL:
neither, and the 95% CI (perm2_ci on a fixed grid [-0.5, 0.5], audit B2) lies within [-0.02, +0.02]. UNMEASURED otherwise.
DIAL summary prints the three labels (no new test).
**M4 / B1.** Every run is re-scored (dropout scale, CPU, same stream); a label that differs under the re-score carries a
FLAG with the re-scored label. Secondaries: wrap-aware opposition (median |wrap(step(+1) + step(-1))| over channels with a
step > 0.05 rad, share below 0.1 rad) beside the raw opposition; accuracy at T = 512 (OOD); floors printed.
**Minor.** The dial also changes the wrap-only revisit share (0.11 / 0.74 / 1.00 at p 0.5 / 0.9 / 1.0; stated); md5 guard adds
data_parallel.py, environment.py, rescore_hook.py; the driver waits for CLOCK_SIDE to finish (shared slots on GPU 0).
Cost: 48 runs x ~5-10 min, 6 at a time on GPU 0: ~1 h; re-score on CPU ~10 min.
