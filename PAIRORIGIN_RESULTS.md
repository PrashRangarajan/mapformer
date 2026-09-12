# PAIRORIGIN -- results (pre-registration `PAIRORIGIN_PREREG.md`, commit e6fc52c)

3 arms x 8 seeds, one batch, standard recency (`k` varying per query). `EMPair_r4` gives MapEM
per-pair position origins -- `q^p_t = p0 + W^q_out W^q_in x_t`, likewise for `k` -- keeping the
Hadamard composition, rank and depth fixed. `W_out` is zero-initialised, so at step 0 it IS
`VanillaEM_P0_r4`. +2,048 parameters (+0.92%).

## The result: P1 CONFIRMED

| arm | acc T=1024 | acc T=2048 | final loss |
|---|---|---|---|
| `Vanilla_r4` (WM) | 0.975 +/- 0.072 | 0.947 | 0.064 |
| **`EMPair_r4`** | **0.880 +/- 0.136** | 0.784 | 0.422 |
| `VanillaEM_P0_r4` | 0.600 +/- 0.126 | 0.510 | 1.340 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EMPair - P0 | **+0.280** | 0.218 | 0.216 | 7/8 | **DETECTABLE** |
| EMPair - P0 at T=2048 | **+0.273** | 0.234 | 0.231 | 7/8 | **DETECTABLE** |
| EMPair - WM | -0.095 | 0.137 | 0.135 | 0/8 | unmeasured |
| P0 - WM (the gap being closed) | -0.375 | 0.156 | 0.154 | 0/8 | DETECTABLE |

- **P1 (kernel sharing): CONFIRMED.** +0.280 >= +0.20 and detectable.
- **P2 (recovery to WM): MET**, in the sense registered -- the residual -0.095 is inside its
  MDE. That is "unmeasured", not "equal": EMPair remains nominally below WM on 0/8 seeds.
- **P3 (per-token search alone): NOT CONFIRMED.** P1 and P3 were mutually exclusive by
  construction, and P1 fired.

## The mechanism, measured on three links of one chain

The intervention changes what it was built to change, what the model then does, and the score --
together:

| readout | `VanillaEM_P0_r4` | `EMPair_r4` | WM |
|---|---|---|---|
| phase spread of the kernel across pairs | 0.000 | **1.448** (seeds 1.28-1.58) | 1.966 |
| solved cells retrieved via a per-token rewind | 0.962 | **0.185** | -- |
| accuracy | 0.600 | 0.880 | 0.975 |

(uniform "no structure" null at this N: 3.274.)

Read together: given per-pair freedom, EM **stops moving the query token** and shapes the kernel
instead. That is the mechanism the sharing account names, and it is now an intervention rather
than a correlation -- the arms are the same function at init and differ in one thing.

Manipulation checks, all three required before the verdicts:
1. Same function at init: max |logit diff| **0.000e+00** -- PASS.
2. Origin pathway moved: 0.327 to 0.841 over 16 tensors -- PASS.
3. Per-pair spread appears: 0.000 -> 1.448 -- PASS (see the probe note below).

## What this does NOT settle

- **Rule 9 bites**: r(final loss, accuracy) = **-0.983** over the 24 runs. Accuracy here is an
  affine readout of training loss, so this is a statement about what training FINDS. That is the
  right frame anyway -- the existence construction already showed representation is not the
  limit (rewind installed and frozen: 1.000, 8/8) -- but a loss-matched residual cannot separate
  the two, because loss-matching conditions on a mediator (`AUDIT_2026-09-10.md` #7).
- **The capacity confound is not controlled.** EMPair adds 2,048 parameters. The clean control is
  per-token origins driven by a CONSTANT (identical parameter count and optimiser treatment, no
  content dependence) -- the `MagOnly` move for this arm. Until that runs, "per-pair freedom" and
  "+0.92% parameters in the position pathway" are not separated. Registered as the next arm.
- **It says nothing about the FIXED-offset case.** `EM_WM_THEORY.md` 2a stands as written: the
  claim that a shared kernel is BETTER when the offset is fixed remains unsupported, and the
  floor-normalised extended-length cell is a separate open question the paper-task rerun tests.
- One task, one architecture, n=8.

## Probe note -- the sixth readout failure of this line, and the second on this arm

Neither probe knew about per-token origins, and both were wrong in different ways:

- `probe_anatomy.em_forward` rebuilt the model from `p0_pos` and so reproduced a DIFFERENT model.
  Its own assert caught it (max diff 13.8) and refused to report -- the failure mode working.
- `probe_phase_spread` branched on the LAYER type, took the EM path, read `p0_pos`, and reported
  a phase spread of exactly **0.000 for the arm whose entire point is per-pair origins**. It
  produced a wrong number, not an error, and the only reason it was caught is that P4 and check 3
  contradicted each other.

Both now route on `hasattr(m, "_origins")` -- where the origins come from -- rather than on the
layer type. Guard tests still pass 13/13. The standing lesson (rule 34) generalises: **a probe
written against one architecture will mis-measure the next one, and the loud failure is the
lucky case.**
