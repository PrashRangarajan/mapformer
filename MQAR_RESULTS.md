# MQAR: not runnable at this project's scale. No batch was launched.

Four pilots, ~15 minutes of GPU. The conclusion is about feasibility, not about
positional encoding, and it is the opposite of what I pre-registered.

## What was predicted, and what happened

`MQAR_PREREG.md` predicted **a ceiling** -- every arm near 1.0, because MQAR exists
to measure how close a sub-quadratic model gets to softmax attention, and every arm
here already is softmax attention.

**It is a floor instead.** Nothing learns the task.

| config | arm | held-out acc | "guess among the episode's values" |
|---|---|---|---|
| n_kv=16, 100 ep | `Vanilla_r4` | 0.075 | 0.063 |
| n_kv=16, 500 ep | `Vanilla_r4` | 0.089 | 0.063 |
| **n_kv=4**, 200 ep | `Vanilla_r4` | **0.273** | **0.250** |
| **n_kv=4**, 200 ep | `RoPE` (index) | **0.264** | 0.250 |
| **n_kv=4**, 200 ep | `PlainFlat` (index) | **0.274** | 0.250 |

Raw chance is 0.0078. Every arm sits at the rate for *knowing which values occur in
this episode and picking among them uniformly*, which the training loss confirms
independently: at n_kv=16 the loss settles at 2.96 against $\ln 16 = 2.77$, and at
n_kv=4 at 1.85 against $\ln 4 = 1.39$. **The models learn the value set and never
the association.** No induction circuit forms.

## Why this is a scale result and not a finding

The decisive check was running the index arms. If path integration were interfering
with content matching -- a plausible story, since a content-dependent rotation gives
the same token a different phase at each occurrence -- then `RoPE` and `PlainFlat`
should have learned where `Vanilla_r4` did not. **They do not: 0.264 and 0.274
against 0.273.** All three fail identically, so the failure is not about the
positional mechanism.

It is capacity and data. MQAR's published setting uses a vocabulary of 8192 and a
hyperparameter sweep of its own; these runs are 2 layers at d=128 over ~150K
sequences. Making it work would be a separate exercise in fitting someone else's
benchmark, not a measurement of our axes.

## What this costs, and what it corrects

**My recommendation to run MQAR was wrong**, and for a reason I should have caught
from the same source I used to justify it: MQAR discriminates **state size** among
models that are *worse* than attention. It has nothing to say about a comparison
between two softmax-attention models that differ only in their positional operator.
That the arms would be indistinguishable was foreseeable; that they would be
indistinguishable **at the floor** was not.

Four pilots caught it for ~15 minutes of GPU, against a 48-run batch that would have
measured noise around 0.25.

## Standing note

Of the field's standard synthetics, **parity** discriminates our axis sharply
(path integration +0.316 at L=16, 8/8 seeds) and **Flip-Flop** and **MQAR** do not
-- Flip-Flop because it only ever asks offset $k=1$, MQAR because it is
content-addressed and aimed at a different architecture class. The anchor for this
work is parity plus the navigation and counting tasks, and adding more borrowed
synthetics is not obviously worth it.
