# Pre-registration: does MapPoPE beat its COMPONENTS on enwik8?

Written before any run in this batch. Closes the open claim in
`ENWIK8_SEEDS.md`, whose own CORRECTION section states the gap and names the
fix: the seeds were on "MapPoPE vs RoPE" while the claim made was "MapPoPE
composes, i.e. beats its components". That is rule 12 -- put the seeds on the
comparison you are CLAIMING.

## Status quo

| arm | val bpc | seeds |
|---|---|---|
| MapPoPE-Flat r4 | 1.3786 | 3 |
| PoPE-Flat | 1.3806 | **1** |
| Vanilla / MapWM r4 | 1.3841 | **1** |
| RoPE | 1.3864 | 3 |

ESTABLISHED: MapPoPE - RoPE = -0.0052, 3/3 seeds.
NOT ESTABLISHED: MapPoPE - PoPE = **-0.0020 at n=1**, which is less than half
PoPE-Flat's own checkpoint sd (0.0051).

## Power, computed BEFORE the batch

Paired sd from the one paired contrast we have (MapPoPE - RoPE, n=3) is
**0.0023**. MDE = 2.8 * sd / sqrt(n):

| n | 3 | 6 | 8 | **12** | 16 |
|---|---|---|---|---|---|
| MDE | 0.0036 | 0.0026 | 0.0022 | **0.0018** | 0.0016 |

**n=3 would return "unmeasured" by construction** against a -0.0020 target, so
running it would be uninformative (rule 11). Required n at this sd is **9.9**.
**This batch commits to n=12 up front.**

Caveat on that sd: it is the MapPoPE-RoPE paired sd. MapPoPE and PoPE share the
PoPE encoding, so their paired sd could be SMALLER (better power than planned)
or larger. n=12 is fixed regardless -- see the stopping rule.

## Stopping rule -- fixed, to avoid sequential testing

**n=12 is decided now and will not be extended on the basis of the result.**
Running until a contrast becomes significant is p-hacking; if n=12 returns
"unmeasured", that is the finding and it will be reported as such. Any later
extension must be registered as a separate batch with its own n.

## Design

- **Arms in the claim, n=12 each, ALL TRAINED IN ONE BATCH**: `MapPoPE-Flat`,
  `PoPE-Flat`, `Vanilla` (all r=4). Retraining every arm together is required:
  the trainer changed since the stored runs (`--data-val`, `--save-ckpt`), and
  comparing fresh arms to stored checkpoints is the error behind the lm200
  retraction (rule 3).
- **`RoPE` at seeds 0-2 as a REPRODUCTION CONTROL.** Stored values are
  1.3864 / 1.3837 / 1.3840. If the fresh runs do not reproduce those, the code
  path changed and the stored enwik8 table is suspect -- which would itself be
  worth knowing.
- Recipe identical to the existing 36k runs: seq 512, batch 16, lr 2e-4,
  dim 512, 9 layers, r=4, 36k iters, deterministic val, enwik8's own hardcoded
  90M/5M split (NO `--data-val`, so the data path is byte-identical to the
  stored runs).
- Readout: mean of the last 5 checkpoints, as in `ENWIK8_SEEDS.md`.

## Predictions

- **E1, PRIMARY.** MapPoPE - PoPE at n=12. `ENWIK8_SEEDS.md`'s own stated prior
  is that this margin is "unlikely to survive". Registered as genuinely open.
- **E2.** MapPoPE - Vanilla (the position component). Currently -0.0056 at n=1,
  i.e. nearly 3x the E1 margin, so it is the likelier of the two to clear.
- **E3.** MapPoPE - RoPE replicates at -0.0052 with 12 seeds.
- **E4, control.** Fresh RoPE seeds 0-2 reproduce the stored values.

**Composition requires BOTH E1 and E2 to clear.** Beating one component is not
composition. If E2 clears and E1 does not, the honest statement is "MapPoPE
beats the position component but is indistinguishable from PoPE alone", which
would mean the enwik8 win is the ENCODING, not the combination -- consistent
with what the code batch found out of distribution, where PoPE - RoPE (-3.585)
dwarfed MapPoPE - PoPE (-0.102).

## What would make this void

- Arms trained in different batches, or against stored checkpoints.
- Runs not converged; all existing enwik8 36k runs are budget-limited (negative
  val slope at 36k), which is carried over here and must be stated beside the
  result. A budget-limited null is weaker than a converged one.
