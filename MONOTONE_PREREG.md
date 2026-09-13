# Pre-registration: the sign ablation outside MapFormer-WM (MONOTONE)

Written 2026-09-13, **before any arm is trained**. Verdict rules below are fixed; whichever
way they point is what goes in `MONOTONE_RESULTS.md`.

## Why

The signed-vs-monotone result (`SIGN_ABLATION.md`, `RECENCY_RESULTS.md`) was an ablation inside
**one** architecture, MapFormer-WM. Two gaps:

1. **MapFormer-EM has never been run monotone.** EM's recency solution is a *subtraction*: the
   query token rewinds theta by -(k-1) symbol steps, found per token and wrapped
   (`SEARCH_RESULTS.md`). "Recency needs only a clock" was shown on WM, which does not have to
   rewind. theta enters through cos/sin, so a forward shift `2 pi n_i / omega_i - (k-1) c_i >= 0`
   is an equivalent per block, and T1 (`THEORY_SEARCH_AND_LENGTH.md`) found good wrapped rewinds
   available almost everywhere. So whether monotone EM can still solve recency is open, and
   I do not register a direction for it.
2. **The sign result has been shown for one generator only.** Selective RoPE's generator
   (`model_selective.py`, `SRoPEGen`: `theta = temp * cumsum(sigmoid(gate x) * conv1d(W_omega x))`)
   is natively signed, is not a MapFormer `ActionToLie`, and has no rank bottleneck, conv-free
   increment or shared-omega readout in common with it.

## The knob

`|.|` on the **final per-channel increment** before the cumsum, nothing else (`model_monotone.py`).
The same placement as `Abs_r4` in `SIGN_ABLATION.md` (after `W_out`; for SRoPE after the gate).

Construction, verified before registration (and re-checked in the batch):
- `EM_P0_Abs_r4` has **every parameter identical** to `VanillaEM_P0_r4` at the same seed and
  consumes the same RNG. A signed twin through the same construction path
  (`EM_P0_Signed_r4`) is bitwise the same function as `VanillaEM_P0_r4` at init (max diff 0.0).
- `SRoPEGen_Abs` has every parameter identical to `SRoPEGen`, the same RNG consumption, and
  238,234 parameters in both. Only `SelectiveAngle.forward` changes.
- Trained increments must be `>= 0` in both constrained arms (manipulation check M-C1).

## Arms, one batch, 12 seeds (0-11)

| id | task | arm | role |
|---|---|---|---|
| E1 | recency | `VanillaEM_P0_r4` | EM signed baseline |
| E2 | recency | `EM_P0_Abs_r4` | **EM monotone, primary** |
| E3 | recency | `Signed_r4` | WM signed |
| E4 | recency | `Abs_r4` | WM monotone (replicates `RECENCY_RESULTS.md` in-batch) |
| E5 | recency | `SRoPEGen` | SRoPE signed |
| E6 | recency | `SRoPEGen_Abs` | SRoPE monotone |
| T1 | torus | `SRoPEGen` | SRoPE signed |
| T2 | torus | `SRoPEGen_Abs` | SRoPE monotone |
| R | torus | `Signed_r4`, `Abs_r4`, seed 0 only | reproduction control vs `runs/sign/p0` |

Recipes are copied, not chosen:
- **recency** = `run_pairorigin.sh`: k_max 64, T 1024, eval T 1024/2048, 300 ep x 48 batches x 16,
  cosine, lr 1e-3, 1 layer, d 128, 2 heads.
- **torus** = `run_sign.sh`: 300 ep x 98 x 128, T=128, cosine, lr 1e-3, held-out map (env-seed
  10000), eval T 128/512/1024, 100 trials.

96 training runs. Accuracy is the held-out accuracy; final loss is the last epoch's training loss.

## Contrasts (all paired by seed; MDE = 2.8 sd / sqrt(n); below MDE = "unmeasured")

**Ceiling clause** (as in `RECENCY_RESULTS.md`): on recency, if both arms of a contrast have mean
>= 0.99 at T=1024, that contrast is read at T=2048 instead.

### Experiment 1 -- does EM need subtraction?

- **P1 (primary)**: `EM_P0_Abs_r4 - VanillaEM_P0_r4`, recency, T=1024.
- **P2**: `Abs_r4 - Signed_r4` (WM), recency, ceiling clause applies.
- **P3**: interaction P1 - P2, paired by seed.
- **P4 (mechanism)**: among solved cells (acc >= 0.9, k >= 8), the fraction retrieved through the
  query token's own step (the `probe_anatomy` rewind route: `max(sel-sel0, selmin-selmin0) >= 0.5`
  in the best head, as in `analyze_pairorigin.py`), per arm, E1 vs E2; plus the number of solved
  cells per seed.

Verdicts:
- **SUBTRACTION NEEDED**: P1 detectably negative AND P3 detectably negative.
- **WRAP SUBSTITUTES**: P1 not detectably negative, MDE(P1) <= 0.15, AND E2's rewind-route
  fraction is >= 0.5 (i.e. monotone EM still solves through the query token's step, necessarily
  forward and wrapped).
- **MONOTONE HELPS EM**: P1 detectably positive.
- Anything else: **UNRESOLVED**, report every number.

Rule 9 is applied to the six recency arms pooled; the loss-matched P1 is printed beside the raw
one. Because loss-matching conditions on a mediator here (a representational limit also shows up
as worse fit), the RAW contrast carries the verdict and the loss-matched one is reported, not read.

### Experiment 2 -- does the sign account transfer to a second generator?

- **Q1 (primary)**: `SRoPEGen_Abs - SRoPEGen`, torus, T=1024, **loss-matched** (pool = the two
  torus arms), because that is the form `SIGN_ABLATION.md`'s -0.280 was read in. Raw printed beside.
- **Q2**: `SRoPEGen_Abs - SRoPEGen`, recency, ceiling clause applies, raw.
- **Q3**: crossover interaction Q1(raw, T=1024) - Q2, unpaired (different tasks).

Predictions from the clock/map account, and their falsifiers:
- Q1 detectably negative. **Falsified** if Q1 loss-matched is not detectably negative with
  MDE <= 0.10.
- Q2 not detectably negative. **Falsified** if Q2 is detectably negative.
- Q3 detectably negative.

Exploratory, not read as verdicts: SRoPE's torus sign cost against MapFormer's stored cost in
`runs/sign` (unpaired), licensed only if R passes; alpha and opposition for the SRoPE arms.

## Manipulation checks, required before any verdict

- **M-C1**: max over a held-out batch of `-min(increment)` is 0 for E2, E6, T2 checkpoints
  (all 12 seeds); signed arms have negative increments.
- **M-C2**: construction identity at init (re-run in the batch, above).
- **M-C3 (R)**: `Signed_r4` and `Abs_r4` seed 0 retrained here are **bitwise** identical to
  `runs/sign/p0`. This is determinism, not replication (rule 27); it only licenses citing the
  stored arms as same-pipeline.

## What this cannot show

- One knob (`|.|`). `softplus`/CARoPE forms carry an init confound and are not run.
- n=12: a first-8 / fresh split is too small to guard against the first-seeds overestimate seen
  four times in this project; effects near the MDE are flagged as such, not called replicated.
- `SRoPEGen` is Selective RoPE's generator in MapFormer's scaffold, not Selective RoPE
  (see `model_selective.py`).
