# The gated signed increment: the separation works and buys nothing

`Delta = sigmoid(W_g x + b) * (W_out W_in x)` -- CoPE's selection on MapFormer's
direction. 4 arms x 8 seeds x 2 tasks, one batch per task, recipes copied verbatim
from `run_sign.sh` and `run_recency.sh`. Pre-registration: `GATED_PREREG.md`.

**The headline is a dissociation.** The gate does exactly what it was designed to
do, verifiably and on every seed -- and it does not improve accuracy anywhere.

## P3 (the mechanism check) -- PASSES, decisively

| arm | gate on ACTIONS | gate on OBSERVATIONS | ratio | per-seed |
|---|---|---|---|---|
| `Gated_r4` | 0.987 ± 0.003 | **0.244 ± 0.043** | **4.16x** | 3.33-5.44, 8/8 |
| `Gated_r2` | 0.943 ± 0.054 | 0.502 ± 0.282 | 2.55x | 0.92-5.30 |
| `Gated_r4_frozen` | 0.982 | 0.982 | **1.00x** | control, exact |

The floor was set in advance from existing data, not chosen: Selective RoPE's gate
measured **1.35x** on this exact contrast and was judged NOT to be suppression.
`Gated_r4` reaches **4.16x** with every seed above 3.3 -- the gate closes to 0.24 on
observations while staying at 0.99 on actions. That is the what/where separation,
explicit and verified. The frozen control reads exactly 1.00x, which is what
licenses reading the rest.

## P1 (torus) -- FAILS

| contrast | T=512 | T=1024 |
|---|---|---|
| **`Gated_r4` - `Vanilla_r4`** | **+0.004** (MDE 0.008, 5/8) | **+0.003** (MDE 0.024, 4/8) |
| `Gated_r4` - `Gated_r4_frozen` | +0.009 (MDE 0.008, 7/8) **DETECTABLE** | +0.009 (MDE 0.023, 6/8) |
| `Gated_r4_frozen` - `Vanilla_r4` | -0.005 (MDE 0.013, 3/8) | -0.006 (MDE 0.024, 3/8) |

**The frozen control earns its place here.** The only detectable contrast is
`Gated_r4` over its frozen twin, and it is almost exactly the size of the frozen
twin's own deficit against the ungated baseline. So what the gate learns is to
undo the constant 0.982 rescale that having a gate imposes. Net against the real
baseline: nothing. Without that arm, `+0.009, 7/8 seeds` would have read as a win.

All arms converged (8/8 flat, final loss 1.4-1.8e-4) except `Gated_r2` (5/8, loss
0.040).

## P2 (recency) -- FAILS, and the direction is negative

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Vanilla_r4` | **0.0119** | 0.9997 | **0.9773 ± 0.015** |
| `Gated_r4_frozen` | 0.0213 | 0.9997 | 0.9397 ± 0.057 |
| `Gated_r4` | 0.0406 | 0.9863 | 0.9384 ± 0.083 |

`Gated_r4 - Vanilla_r4` is **-0.039** raw (MDE 0.079, 3/8) and **-0.016**
loss-matched (MDE 0.055) -- unmeasured in both, never positive. This is the
sharpest failure of the prediction, because recency is the task where the recency
ablation showed a gate is worth **+0.594** when supplied by intervention. Supplying
it by construction is worth nothing. The gate also *fits worse* (0.0406 against
0.0119), so it costs optimisation.

## P4 -- NOT TESTABLE AS RUN. My design error.

P4 asked whether the gate helps `r=2` more than `r=4`, which needs
`Gated_r2 - Vanilla_r2` against `Gated_r4 - Vanilla_r4`. **`Vanilla_r2` was not in
the batch.** The available contrast, `Gated_r2 - Vanilla_r4` (+0.020 at T=2048,
7/8, detectable), confounds the gate with the rank and answers nothing. Comparing
against the published `r=2` numbers would break rule 3. Unresolved.

## What this means

**MapFormer's linear bottleneck was already doing the separation well enough.** The
gate reaches 4.16x action-vs-observation separation; a trained ungated model already
moves position about five times more on actions than observations. The explicit gate
reproduces a separation that was already there, at 258 extra parameters and a worse
fit.

That is a direct corroboration of the paper's design choice rather than an
improvement on it. MapFormer's stated purpose for the two-stage projection is that
"the model has to identify that action tokens update the internal position, while
ensuring that observations leave the structure untouched" -- and the measurement
says it identifies them well enough that helping is not worth anything.

**The review's borrow recommendation is withdrawn on the evidence.** Section
`sec:borrow` argued from the recency ablation that supplying the gate explicitly
should be free or better. It is neither. What the recency ablation established was
that the gate is *load-bearing* -- removing it destroys the task -- and that does
not imply the model needs help building one. Those are different claims and I ran
them together.

## Honest caveats

- Power: seed sds are 0.008-0.083 against effects of 0.003-0.039, so small real
  effects would not be visible. **Unmeasured, not null** (rule 11).
- r(final loss, accuracy) is -0.999 / -0.734 / -0.702 on the torus at T=128/512/1024
  and -0.949 / -0.657 on recency, so the recency contrast is reported loss-matched
  as well as raw.
- One gate granularity (per token, per head), one init (bias 4.0), one gate shape
  (sigmoid on the increment, multiplicative). A gate on the *bottleneck input*
  rather than the output, or a per-block gate, is untested.
