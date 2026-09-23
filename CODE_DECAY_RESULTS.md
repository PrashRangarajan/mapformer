# C2, the repaired-baseline control: in distribution the envelope HELPS, and both axes are detectably WORSE

Registered in `CODE_PREREG.md` Amendment 2 as "the decisive one". Batch completed
2026-09-22 02:52 (`runs/code_decay/.done`), 4 arms x 3 seeds in one batch, and sat
**unread for a day** until an audit found it. Arms verified as exact inert twins of
their bases before launch (maxdiff 0.00e+00 with lambda -> 0).

## In distribution (val bpc at 512, lower better)

| arm | with decay | without (`runs/code`) | effect of the envelope |
|---|---|---|---|
| **RoPE-Decay** | **0.8967** | 0.9097 | **-0.0130** (MDE 0.0110, 3/3) |
| MapWM-Decay | 0.9000 | 0.9191 | **-0.0191** (MDE 0.0075, 3/3) |
| PoPE-Decay | 0.9046 | 0.9131 | -0.0085 (MDE 0.0091, 3/3) |
| MapPoPE-Decay | 0.9108 | 0.9179 | **-0.0071** (MDE 0.0030, 3/3) |

**C2c's registered expectation is INVERTED.** The envelope was expected to COST in
distribution -- on Bach it did, +0.018 to +0.026, 0/5 seeds better. Here it HELPS
every arm, 3/3 seeds each, by 0.007-0.019 bpc against a measured cross-batch
reproducibility floor of 0.0021/0.0028. Those deltas are cross-batch, so provisional
at the floor's own scale, but the sign is 12/12 seeds.

**A plain index model with 48 extra parameters is the best of all eight code arms
measured at 512.** RoPE-Decay 0.8967 beats every arm in the 2x2, decayed or not.

## The two contrasts C2a registered, with the baselines repaired

- **PoPE-Decay - RoPE-Decay = +0.0079** (MDE 0.0053, **0/3** seeds better)
  **DETECTABLE AGAINST PoPE.** The encoding is detectably WORSE once both arms get
  the envelope.
- **MapPoPE-Decay - PoPE-Decay = +0.0063** (MDE 0.0035, **0/3**)
  **DETECTABLE AGAINST path integration** on the PoPE row.

Ordering with repaired baselines: **RoPE < MapWM < PoPE < MapPoPE**, i.e. exactly
the reverse of the retracted out-of-distribution story, and now with both axes
detectably costing rather than merely unmeasured.

## What this does to C2a

C2a asked whether MapPoPE stays ahead once the baselines are repaired. **In
distribution it is last of four and detectably behind.** The OOD half still needs
`eval_code_long.py` on these checkpoints (a GPU eval, not yet run) -- but note the
retracted OOD framing is what C2a was defending, and C1 already killed it at matched
training length.

## Caveats

- Cross-batch comparison for the "effect of the envelope" column (`runs/code_decay`
  vs `runs/code`); the within-batch contrasts are clean.
- Budget-limited like every code arm here: negative val slope at 36k, constant LR,
  no warmup, no decay, no weight decay.
- `best_val_bpc` is a min over 36 evaluations of 40 windows (~5.7% of the val file),
  so it carries min-selection bias; the full-val readout is the better one and was
  used for the ablation.
