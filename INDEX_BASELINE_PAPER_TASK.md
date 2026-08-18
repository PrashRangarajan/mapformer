# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper reports MapFormer-WM **0.955**, MapFormer-EM **0.999**.

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.989 ± 0.010 |
| VanillaEM_P0 | 0.987 ± 0.012 | 0.987 ± 0.012 |
| RoPE | 0.510 ± 0.005 | 0.516 ± 0.004 |
| PlainFlat | 0.509 ± 0.019 | 0.513 ± 0.012 |

## What this was run to settle

`PAPER_TASK_ABLATION.md` showed the trained models USE the action stream, but an
ablation cannot say whether a model COULD solve the task without path
integration. That needs a model that never had it. Same recipe, same
hyperparameters, same seeds as the rows above -- only the position code differs,
with parameters matched to 0.2% (Vanilla 204,373 / RoPE 203,925 /
PlainFlat 203,925).

**`RoPE` is the tight control**: MapFormer-WM with `theta = t * freqs` instead of
`omega * cumsum(Delta)`. Identical architecture, position code swapped.
`PlainFlat` is an ordinary transformer with index RoPE -- the same control used
in Match-Query, included so the two tasks are judged against the same baseline.

### The floor, measured not assumed

The paper's task scores every revisited observation including blanks, and
p_empty=0.5, so the operative floor is the always-predict-blank rate. Measured on
these events: **0.506** (`PAPER_TASK_ABLATION.json`). It is NOT 1/16.

| variant | position code | fresh-map acc | vs 0.506 floor |
|---|---|---|---|
| Vanilla | path-integrated | 0.989 ± 0.010 | +0.483 |
| VanillaEM_P0 | path-integrated | 0.987 ± 0.012 | +0.481 |
| RoPE | index | 0.516 ± 0.004 | **+0.010** |
| PlainFlat | index | 0.513 ± 0.012 | **+0.007** |

**Index-position models do not solve the paper's task at all.** They sit at the
always-predict-blank floor, within noise of it, on 3/3 seeds each.

### Three consequences

**1. It settles the question the ablation could not.** The content route --
~4-5 revealed observations carry enough bits to pin a cell on a 4096-cell torus
-- exists information-theoretically but is not usable by these models. It is not
that they *choose* path integration; without it they are at the floor.

**2. My "Match-Query closes a route the paper's task leaves open" claim is not
merely unsupported, it is false.** There was no open route to close. The
withdrawal in `PAPER_TASK_ABLATION.md` stands, and the reason is now stronger
than "unmeasured".

**3. It strengthens the project's headline result by moving it onto the paper's
own task.** "Path integration is necessary for in-context cognitive maps" was
carried almost entirely by Match-Query, a task invented here. It now holds on the
paper's task too, at the paper's own configuration and against the paper's own
metric: 0.989 with path integration, 0.513 without. That directly answers the
standing caveat that one new task was carrying the claim.

### Loose thread, flagged not resolved

Final training loss for the index models is 1.59-1.68 nats, below the marginal
entropy of the observation distribution (2.079 nats for blank at 0.5 plus 16
uniform types). So they beat the marginal in LIKELIHOOD while sitting exactly at
it in ACCURACY.

One untested hypothesis: the walk is directed with run lengths 1-10, so an
out-and-back run retraces cells in reverse order, and that is detectable from the
ACTION TOKENS AS CONTENT without any position code. That would give short-range
retrace gains without a map. It is a guess and has not been measured; the way to
check is per-revisit-distance accuracy (how many steps ago the cell was last
seen), which would show index models winning only at very short recurrence
intervals.
