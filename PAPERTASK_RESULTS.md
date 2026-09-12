# PAPERTASK rerun -- results (pre-registration `PAPERTASK_PREREG.md`, commit 415e96e)

3 arms x 8 seeds, one batch, trained fresh at **50 epochs cosine** (the original was 16 on the
LinearLR default) with the **training logs kept**. Floors measured per condition
(`PAPER_TASK_FLOORS.md`); the primary readout is floor-normalised accuracy, because at pe=0.8
four fifths of the raw scale is floor.

## The convergence gate FAILED, so P1 is not read

| arm | IID | >= 0.99 required | mean final training loss |
|---|---|---|---|
| `Vanilla` (WM) | **0.968** | no | 0.1306 |
| `VanillaEM_P0` | **0.985** | no | 0.0762 |
| `MapPoPE-Flat` | 1.000 | yes | 0.0168 |

The pre-registration says: no verdict is read until every arm reaches IID >= 0.99. It did not
happen, so **P1 is NOT READ** and **P4 does not fire**.

**And the gate was probably mis-set.** WM scores 0.969 at 16 epochs on LinearLR and **0.968 at
50 epochs on cosine** -- 3x the budget and a better schedule moved it by 0.001. Its shortfall
against the paper's 0.99 is SYSTEMATIC, not a budget artifact, so a gate demanding 0.99 from WM
on this task could probably never pass. That is the "check whether a verdict cell could have
gone the other way" error (rule 11's corollary), committed again in the arm I wrote to avoid it.

## What the numbers say, reported but not interpreted as a verdict

Floor-normalised (fraction of the available range used), paired by seed:

| contrast | delta | MDE | seeds + | verdict | same contrast at 16 epochs |
|---|---|---|---|---|---|
| EM - WM, l=512 | +0.067 | 0.168 | 4/8 | unmeasured | +0.174 |
| EM - WM, l=1024 | **+0.186** | 0.185 | 8/8 | DETECTABLE | +0.352 |
| EM - WM, l=2048 | **+0.287** | 0.194 | 8/8 | DETECTABLE | +0.430 |
| MapPoPE - EM, l=512 | +0.066 | 0.079 | 6/8 | unmeasured | +0.070 |
| MapPoPE - EM, l=1024 | +0.073 | 0.090 | 6/8 | unmeasured | +0.112 |
| MapPoPE - EM, l=2048 | +0.102 | 0.128 | 6/8 | unmeasured | +0.159 |

- The extended-length EM advantage **survives a 3x budget and a better schedule at about
  two thirds of its size**, still 8/8 at both extended lengths.
- **P3 holds as predicted**: MapPoPE - EM stays unmeasured at every length. The per-pair-kernel
  arm remains directionally the best model on this task and still cannot be shown to beat EM.

## Rule 9 -- the check the deleted logs made impossible, now run

r(final loss, accuracy) = **-0.461** over the 24 runs (acc = 0.942 - 0.152*loss, resid sd 0.036).

This is the interesting one. Every contrast in the recency line runs at |r| = 0.93-0.99, where
accuracy is an affine readout of training loss. **On the paper task it is not** -- so the
extended-length contrast is not simply a loss gap, and the loss-matched objection that hangs
over the recency results does not apply here. Note also that WM has the WORST training loss by
~2x while being the arm that degrades with length, which is consistent with its IID shortfall
being real rather than a measurement artifact.

## Where this leaves `EM_WM_THEORY.md` 2a

2a demoted "when the offset is fixed, a shared kernel is better and EM wins" on four grounds.
This rerun settles two of them and leaves two standing:

- **Settled: not a budget/recipe artifact.** 3x budget, cosine, logs kept: the effect shrinks by
  a third and stays detectable at both extended lengths, and rule 9 says it is not a loss gap.
- **Settled: not a floor artifact** (already, by the measured floors -- normalising made it
  larger, not smaller).
- **Still standing: the axis is LENGTH, not offset-fixedness.** +0.067 / +0.186 / +0.287 is
  monotone in length, and "helps at OOD length" is this project's universal unexplained
  signature.
- **Still standing: the counterexamples on other fixed-offset tasks** (MiniGrid, vocab,
  Match-Query, compositional, family tree).

So the honest statement is unchanged in shape and firmer in evidence: **EM degrades with length
more slowly than WM on this task, robustly; that is not the same claim as "a shared kernel is
better when the offset is fixed", and nothing here supports the second.**
