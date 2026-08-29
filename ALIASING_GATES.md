## grid 32, n_obs=16

# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=32  (T=512, 2015 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.481 (blank 0.48), non-blank=0.077 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.256 |
| G4 action n-gram | PASS | o1..5 0.483/0.481/0.430/0.413/0.423 (best 0.483 vs marg 0.481, non-blank best 0.055 vs 0.077) |
| G5 label mass | PASS | 0.098 of steps scored; 50.4/traj [25..81] |
| G6 revisit lag | PASS | median=33 p90=244 max=491; within8/16/32=0.01/0.27/0.50 |
| G7 oracle | PASS | 1.0000 over 2015 scored |
| G8 vocab range | PASS | tokens [0, 25] vs vocab 26 |


## grid 32, n_obs=64

# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=64 (chance 1/16 = 0.0156), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=32  (T=512, 2015 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0156 = 1/64 |
| G2 marginal | PASS | all=0.481 (blank 0.48), non-blank=0.031 vs chance 0.016 |
| G3 copy-last-obs | PASS | 0.244 |
| G4 action n-gram | PASS | o1..5 0.483/0.481/0.425/0.408/0.420 (best 0.483 vs marg 0.481, non-blank best 0.019 vs 0.031) |
| G5 label mass | PASS | 0.098 of steps scored; 50.4/traj [25..81] |
| G6 revisit lag | PASS | median=33 p90=244 max=491; within8/16/32=0.01/0.27/0.50 |
| G7 oracle | PASS | 1.0000 over 2015 scored |
| G8 vocab range | PASS | tokens [0, 73] vs vocab 74 |


## grid 32, n_obs=256

# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=256 (chance 1/16 = 0.0039), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=32  (T=512, 2015 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0039 = 1/256 |
| G2 marginal | WARN | all=0.481 (blank 0.48), non-blank=0.013 vs chance 0.004 |
| G3 copy-last-obs | PASS | 0.238 |
| G4 action n-gram | PASS | o1..5 0.483/0.481/0.425/0.408/0.420 (best 0.483 vs marg 0.481, non-blank best 0.000 vs 0.013) |
| G5 label mass | PASS | 0.098 of steps scored; 50.4/traj [25..81] |
| G6 revisit lag | PASS | median=33 p90=244 max=491; within8/16/32=0.01/0.27/0.50 |
| G7 oracle | PASS | 1.0000 over 2015 scored |
| G8 vocab range | PASS | tokens [0, 265] vs vocab 266 |



---

## Interim note (2026-08-29 05:40, 6 of 25 runs) -- NOT a result

Recorded while the sweep runs so the decision trail is not reconstructed later.

n_obs=256 seeds 0-2 finished. Raw paired effect **+0.389** (+0.394/+0.364/+0.408),
i.e. more than DOUBLE the +0.173 anchor at n_obs=16 -- the opposite direction from
pre-registered outcome A, which predicted a collapse toward 0 at low aliasing.

**It is not interpretable yet.** The index arm is NOT converged in 2 of 3 seeds
(slope -0.00074 / -0.00060 against a 5e-4 flat threshold, still dropping ~0.02-0.03
over the final 40 epochs) while the path-integrated arm is flat at loss 0.007-0.029.
An unconverged index arm inflates the effect by exactly the time-to-solve confound
that inverted the grid-8 sign.

Outcome C (floor collapse) IS ruled out: RoPE sits at 0.60, far above the 0.013
non-blank marginal. Both arms are learning.

**The design tension this exposes.** The same 400-epoch budget converges the index
arm at n_obs=16 (flat 3/3, loss 0.41-0.50) but not at n_obs=256. Harder conditions
need longer budgets, so "matched budget across conditions" and "converged within
each condition" cannot both hold. Options, none chosen yet:
  1. All conditions to 800 ep -- fully matched AND converged, ~55 h.
  2. Each condition to its own convergence -- defensible, but then cross-condition
     effect comparison is budget-confounded, which is the weak link.
  3. Report at matched 400 ep with the non-convergence flagged, and add an
     extended-budget arm at n_obs=256 to BOUND how much of the effect is budget.

Deciding on 3 of 25 runs would be the same mistake as fitting a trend through two
budget points (rule 5 corollary). Revisit when the sweep completes.

Power note: the n_obs=256 effect has sd 0.022 at n=3 (MDE 0.036), far tighter than
the n_obs=16 anchor's sd 0.125. The basin-selection variance that forced n=5 is
specific to the anchor, not general.
