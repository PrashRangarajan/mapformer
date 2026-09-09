# EM's collapse was the origin vector, and fixing it makes EM match WM

5 arms x 8 seeds, ONE batch, allocentric DoorKey-16x16, same recipe as
`MINIGRID_EM.md`. Parameter spread 0.08%. Pre-registration:
`MINIGRID_EM_FIX_PREREG.md`. Floor at T=1024: 0.490.

## P1 -- the shape, reported before any mean, as fixed in advance

| arm | fixes | **min** | 2nd | **gap** | sd |
|---|---|---|---|---|---|
| `Vanilla_r4` (WM) | -- | 0.798 | 0.814 | **0.016** | 0.014 |
| `VanillaEM` | neither | 0.602 | 0.739 | **0.137** | 0.073 |
| `VanillaEM_r4` | rank only | 0.712 | 0.810 | **0.098** | 0.040 |
| `VanillaEM_P0` | origin only | 0.797 | 0.813 | **0.016** | 0.014 |
| **`VanillaEM_P0_r4`** | **both** | **0.816** | 0.818 | **0.002** | **0.011** |

**The collapse is gone, and the origin vector is what did it.** The
worst-to-second-worst gap runs `0.137 -> 0.098 -> 0.016 -> 0.002`: `r=4` alone barely
dents it, shared `p_0` alone brings it to WM's level, and both together give the
tightest distribution in the batch. `VanillaEM_P0_r4`'s spread (sd 0.011) and floor
(min 0.816) are **better than WM's** (0.014, 0.798).

## P2 -- the mean gain was small, and that was the prediction

Predicted `~ +0.019`: the collapse's own contribution, since the trimmed means were
already equal. Measured `EM_P0_r4 - EM_r4`:

| | T=512 | T=1024 |
|---|---|---|
| measured | +0.0139 | **+0.0212** |
| predicted | -- | **+0.019** |

Unmeasured at n=8 (MDE 0.044), but the point estimate lands on the prediction.
**Shared `p_0` does nothing beyond preventing the collapse** -- which is what makes
it a fix rather than an improvement, and is the claim I would have had to withdraw
had the gain been larger.

## P3 -- the two pathologies compound, as predicted

Shared `p_0` buys `+0.050` at `r=2` and `+0.021` at `r=4` (T=1024): interaction
`-0.029`, direction as pre-registered, unmeasured (MDE 0.104). A skewed basis and an
unpeaked `A_P` are separate failures, and fixing one leaves less for the other.

## The answer to the question that started this

| contrast | T=512 | T=1024 |
|---|---|---|
| `EM_P0_r4 - Vanilla_r4` | +0.0012 (4/8) | +0.0035 (6/8) |

**Unmeasured -- i.e. EM MATCHES WM once its initialisation is fixed.** Not better,
not worse. `VanillaEM_P0_r4` is nonetheless the arm to prefer here: highest mean at
both lengths (0.847 / 0.830), tightest spread, and the highest worst-seed of
anything measured.

## What this puts in question

**Every "EM is worse" result in this project used the arm with the pathology.** The
vocab sweep, Match-Query and compositional all ran `VanillaEM` -- paper-faithful
separate `q0/k0`. On this task that arm's failure is entirely an init artefact.

It is not a universal fix, and that must be said: on the torus vocab sweep at
`n_obs=256`, `VanillaEM_P0` still had a collapsed seed (per-seed 0.910 / 0.502 /
0.906). Shared `p_0` removed the collapse **here**, at `r=2` and `r=4`; it did not
there, at `r=2`. `EM_P0_r4` has never been run on that task, and is the obvious
re-test.

## Caveats

Every contrast is unmeasured at n=8 -- the gap and sd columns are descriptive, not
tests, though the ordering `0.137 / 0.098 / 0.016 / 0.002` is unambiguous. One
environment, one tokenization, flat models, 50 epochs. The paper's separate `q0/k0`
is App. A.4's stated design ("we suspect this separation to be beneficial"); this is
the fourth measurement in this project refuting that suspicion, after +0.089 on the
paper task, +0.167 compositional and +0.358 on Match-Query.
