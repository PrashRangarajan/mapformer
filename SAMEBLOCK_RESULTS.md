# SAMEBLOCK results: position mechanisms for addition inside one block

Pre-registration: `SAMEBLOCK_PREREG.md` (165fdc1), amendment 1 (4030e57), compile check `SAMEBLOCK_COMPILE_CHECK.md`
(pass). Mechanical report: `SAMEBLOCK_RAW.md` (`analyze_sameblock.py`).

Cho et al.'s block and recipe throughout. Trained on 1-30-digit additions; exact match on 512 problems per length.
Seed 0 used the original code; seeds 1-2 used the validated compiled path with the vectorised generator. Every
paired comparison is within one code path.

## Results

| arm | format | seeds | exact 30 (training) | exact 60 | exact 100 | exact 150 | per-digit at 60 |
|---|---|---|---|---|---|---|---|
| `ChoPos_coupled` (oracle) | role | 0 / 1 / 2 | 1.000 / 1.000 / 1.000 | 0.998 / 1.000 / 0.982 | **0.451 / 0.801 / 0.838** | 0.000 / 0.043 / 0.057 | 1.000 |
| `ChoPos_coupled` (oracle) | shared | 0 | 1.000 | 1.000 | **0.988** | 0.590 | 1.000 |
| `ChoPos_signed` (MapFormer) | role | 0 / 1 / 2 | **1.000 / 1.000 / 1.000** | **0.000 / 0.000 / 0.000** | 0.000 | 0.000 | 0.163 / 0.354 / 0.689 |
| `ChoPos_abs` (monotone) | role | 0 / 1 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.096 / 0.106 |
| `ChoPos_rope` (index) | role | 0 / 1 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.096 / 0.100 |
| `ChoPos_nope` | role | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 |

## Registered verdicts

- **G1 (control): FAIL.** The role-format oracle averages 0.697 exact match at 100 digits (0.451, 0.801, 0.838),
  below the pre-registered 0.9. The shared-format oracle (seed 0) reaches 0.988. By the pre-registration, P1 and
  P2 are **not read**.
- **G2 (learned the task):**
  - `ChoPos_signed` passes on 3/3 seeds and `ChoPos_coupled` on 3/3.
  - `ChoPos_abs` fails 2/2, `ChoPos_rope` fails 2/2 and `ChoPos_nope` fails 1/1. These arms did not learn
    30-digit addition in this block.
- **P1, P2:** not read (G1). Descriptively they are undefined anyway: the monotone and index arms never learned the
  task, so the contrasts compare zeros.
- **P3:** dropped by amendment 1.

## What the numbers show (descriptive; not verdicts)

- **In this block, no learned position code length-generalises.**
  - Signed MapFormer learns addition perfectly at training length (3/3 seeds) but scores 0.000 exact match at
    60 digits on every seed.
  - Its per-digit accuracy at 60 digits is 0.16-0.69, so it partly works but is nowhere near a whole
    correct sum.
- **Only signed MapFormer, among the learned codes, learns the task at all.** Monotone increments, index RoPE and
  NoPE do not, in this block, at this learning rate.
- **The learned increments have the predicted sign structure on every signed seed.**
  - In each head the sum-digit increment points against the first operand's (cosine -0.48 to -0.92) and against
    the second operand's (-0.38 to -0.79).
  - The monotone arm's cosines are all positive (+0.42 to +0.74), as they must be.
  - The coupling pattern is therefore learned. It is not sufficient for length generalisation here.
- **The pilot-2 hint does not replicate.** In this repo's own layer, one seed, signed MapFormer reached 0.79 at 60
  digits (`ADDITION_PILOT2.md`). In Cho et al.'s block it reaches 0.000 on 3/3 seeds. The architecture and learning
  rate changed together, so which one matters is unknown.
- **Role tags hurt the oracle's generalisation** (0.697 mean at 100 digits against 0.988 shared). Tagging digits by
  role with separate token copies is not free for a coupled-position model.

## Caveats

- **Seeds and code path.** n = 1-3 per arm; seed 0 on the original code, seeds 1-2 on the compiled path.
- **One architecture and one recipe,** chosen because the oracle works there, not because MapFormer does. The
  repo-layer result (pilot 2) came from a different learning rate and block.
- **No eval between 30 and 60 digits,** so whether signed MapFormer degrades gradually or falls off a cliff just
  past 30 digits is unknown.

## Where this leaves the addition line

The pre-registered question cannot be answered here: the control failed in the role format, and neither the
monotone nor the index arm learned the task. The strongest descriptive statement is narrower.

In Cho et al.'s block, MapFormer's path integration learns the coupling pattern and solves addition at training
length, but does not carry it to 2x the length.

Cheap follow-ups:
1. **Evaluation-only, minutes:** evaluate the trained signed checkpoints at 32-58 digits to locate where they break.
2. **Replicate the pilot-2 hint** (repo layer, lr 1e-3, signed and oracle, 3 seeds) to see whether the architecture
   is what made signed MapFormer generalise there.

## Follow-up 1 (evaluation only, run after the results above): where signed MapFormer breaks

Role format, the same checkpoints, 256 problems per length; cells are exact match (per-digit accuracy).

| run | 30 | 31 | 32 | 33 | 35 | 38 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| signed s0 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.98 (1.00) | 0.91 (1.00) | 0.63 (0.99) | 0.01 (0.90) | 0.00 (0.60) |
| signed s1 | 1.00 (1.00) | 0.99 (1.00) | 0.96 (1.00) | 0.96 (1.00) | 0.89 (1.00) | 0.70 (0.99) | 0.29 (0.97) | 0.00 (0.88) | 0.00 (0.73) |
| signed s2 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.99 (1.00) | 0.48 (0.98) | 0.36 (0.98) | 0.25 (0.98) | 0.07 (0.96) | 0.00 (0.89) |
| coupled s0-s2 | 1.00 (1.00) at every length, all three seeds | | | | | | | | |

**Reading.**
- **Signed MapFormer does generalise a little, not at all is wrong.** It holds 0.9 or better exact match up to
  about 33-38 digits, depending on the seed, then falls to 0 by 45-50.
- **It degrades gradually per digit.** Per-digit accuracy is still 0.88-0.96 at 45 digits, where exact match is
  already near 0. The whole-number failure comes from small per-digit errors compounding.
- **Scale of the generalisation.** Signed MapFormer reaches about 1.1-1.3x the training length. The oracle is
  perfect through 50 digits on all seeds, and its role-format failures start between 60 and 100.
- **The earlier wording "scores 0.000 at 60" is correct, but it hid this gradual profile.**
