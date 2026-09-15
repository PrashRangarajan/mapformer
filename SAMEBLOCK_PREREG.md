# SAMEBLOCK: position mechanisms for multi-digit addition inside one block

Pre-registered 2026-09-15, before any arm is trained.

## Why

- **The control.** `ADDITION_CHO_REPRO.md` showed the position-coupling control generalises cleanly to
  100-digit addition (0.938 exact match, 3.3x the training length) in Cho et al.'s block and recipe.
- **The problem with earlier comparisons.** MapFormer's arms used this repo's layer, which does not
  train under that recipe.
- **The fix here.** Every position mechanism is placed inside the SAME block (`model_cho_positions.py`),
  so arms differ only in how position enters attention.

Verified before launch:
- `ChoPos_coupled` is bitwise the same function as `ChoCoupledAPE` given the same weights.
- `ChoPos_signed` with zero increments is bitwise `ChoPos_nope`.

## The question

With digit roles marked, does a learned path-integrated position code discover digit alignment that
length-generalises?
- Is it better with signed increments than monotone ones?
- Is it better than an index code?

Background, from `ADDITION_DESIGN.md`:
- Role-tagged digits let a signed increment express exact position coupling; a monotone one or an
  index code cannot.
- With shared digit tokens, no token-driven code can.

## Arms and seeds

| arm | format | seeds | role |
|---|---|---|---|
| `ChoPos_signed` | role | 0, 1, 2 | the question: MapFormer's path integration, rank 4 |
| `ChoPos_abs` | role | 0, 1, 2 | sign control: monotone increments |
| `ChoPos_rope` | role | 0, 1, 2 | index code |
| `ChoPos_coupled` | role | 0, 1, 2 | oracle / positive control |
| `ChoPos_nope` | role | 0 | no position (literature baseline) |
| `ChoPos_coupled` | shared | 0 | oracle, literature format |
| `ChoPos_signed` | shared | 0 | predicted unable to couple |

## Recipe (Cho et al., Table 1; identical for every arm)

- 1 layer, 4 heads, d = 512;
- Adam, lr 1e-4, 1% warmup, cosine decay to 0.1 lr;
- 50,000 steps x batch 1,000; max_pos 202;
- trained on 1-30 digits with balanced sampling;
- bfloat16 autocast for every arm.

Evaluation: exact match and per-digit accuracy at 30, 60, 100 and 150 digits, 512 problems each.

## Gates, checked before any contrast is read

- **G1 (control).** `ChoPos_coupled` (role) mean exact match at 100 digits >= 0.9. If it fails, no contrast
  is read.
- **G2 (trained).** An arm-seed enters a contrast only if its exact match at 30 digits (training length)
  is >= 0.9. Arms that fail G2 are reported as "did not learn the task".

## Primary readout and predictions (n=3: exploratory by the project's convention; MDEs reported)

Primary readout: exact match at 100 digits, role format, paired by seed.

- **P1.** `ChoPos_signed` - `ChoPos_abs` > 0.
- **P2.** `ChoPos_signed` - `ChoPos_rope` > 0.
- **P3.** Shared format: `ChoPos_signed` exact match at 60 digits < 0.1, where the shared `ChoPos_coupled`
  (seed 0) is expected to pass.
- **No prediction** for `ChoPos_signed` against `ChoPos_coupled`.

## Mechanism readout (descriptive)

For each signed and monotone checkpoint, the per-head cosine between mean role increments (sum against first
operand, sum against second operand, first against second), as in `ADDITION_PILOT.md`.

## Amendment 1 (2026-09-15, written while seed 0 was still training; no final evaluation had been read)

The user asked for a shorter batch. After seed 0 finishes, the queue is trimmed by rules fixed here. The only
information available when these rules were written was the seed-0 runs' in-training exact match at 30 digits.

- **Seeds 1-2 always run** for `ChoPos_signed` and `ChoPos_coupled` (role).
- **Other role arms** (`ChoPos_abs`, `ChoPos_rope`):
  - if seed 0 passes G2 (exact match at 30 digits >= 0.9), seeds 1-2 run;
  - if it fails G2, seed 1 only runs, to check the failure is not seed-specific. The arm is then reported
    as "did not learn the task" if both seeds fail.
- **`ChoPos_nope`** (role) stays at seed 0.
- **`ChoPos_signed` (shared)** is dropped. P3 is answered only descriptively, from the pilots.
- **Seeds 1-2 use the faster code** (`--compile`, which implies the vectorised generator) only if
  `verify_addition_compile.py` shows, for every arm:
  - median relative loss difference between eager and compiled training on identical batches below 0.01;
  - final losses within 5% (or both below 1e-3);
  - a speed-up of at least 1.2x.

  Otherwise they use the seed-0 code. Whichever is used, seeds run on the new code are marked in the results,
  and the P1/P2 contrasts remain paired within seed, so both arms of every pair share one code path.
