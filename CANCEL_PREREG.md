# H3, the cancellation knob -- pre-registration (2026-09-27, before any run of the batch)

## Question
`project_clock_vs_map.md`: signed increments make a map, monotone ones a clock. Dyck's surviving
result (`DYCK_MDEPTH_RESULTS.md`) is depth-substitution: one layer of path integration is worth ~3
layers of attention. Cross-task transfers of such effects have failed repeatedly in this project.
So vary, inside ONE task, the property path integration is supposed to exploit -- how often the
walk's increments cancel -- and read the depth-substitution exchange rate as a function of it.
Is "map vs clock" a dichotomy (any cancellation breaks index attention as badly as full
cancellation) or a continuum?

## Task (`environment_cancel.py`, gated `docs/audits/2026-09-27/gate_cancel.py`)
A 32-cell ring, observation map REDRAWN per trajectory (K=16, p_empty 0.5), GridWorld's directed
walk (action repeated k ~ U{1..10}), action +1 with probability `p_plus`, else -1. Interleaved
`[a, o, ...]`, loss on revisited observations, T=128 steps. Gate: revisit rate 0.758 / 0.751 /
0.750 / 0.750 at p_plus 0.5 / 0.75 / 0.9 / 1.0 (the knob does not change how much there is to
retrieve). Floor, the better of best constant and an n-gram (orders 1-5) over preceding tokens:
**0.554 / 0.544 / 0.520 / 0.498**. A fixed-lag copy (the observation exactly one lap back) scores
0.284 / 0.375 / 0.626 / 1.000 -- the knob moves the task from map to clock.
Caveat fixed now: at p_plus = 1 the action token is constant and path integration's phase is a
LEARNED multiple of the index, not the index; a residual path advantage there is a learned-frequency
effect, not a map effect.

## Pilot (`runs/cancel_pilot`, 2 seeds, 300 epochs; NOT reused) -- sets the recipe and the scale
T=128 held-out accuracy: path 1 layer 1.000 at p 0.5 and 1.0; index 1 layer 0.739 / 0.740 at p 0.5,
1.000 at p 1.0; index 4 layers 1.000 at p 0.5 (final loss 0.010-0.014, still descending slowly).
Index 1 layer at p 0.5 is STALLED-to-slowly-descending at 300 epochs (loss 1.056 -> 1.027 over the
last 100): every claim below is budget-scoped to 300 epochs.

## Batch (one batch, 8 seeds, all arms retrained; `run_cancel.sh`, `train_cancel.py`)
p_plus in {0.5, 0.75, 0.9, 1.0} x arms {path 1 layer (`Vanilla`), index 1 / 2 / 3 layers (`RoPE`)}
= 16 cells x 8 seeds = 128 runs. Recipe: batch 128, T=128, 300 epochs x 98 batches, lr 1e-3, warmup +
cosine, d 128, 2 heads, `--data-workers 3`. Readout: revisit accuracy at T=128 (the training
length; T=512 is printed but is extrapolation and carries no verdict), 200 trials, eval RNG fixed.

## Readouts and branches
Primary: index 1-layer accuracy a1(p) at each p; gap G(p) = path1(p) - a1(p). Pairwise exact
permutation tests (`stats_core.perm2_p`) between adjacent and end points; "differs" = p < 0.05.
- **CONTINUUM** -- a1(0.75) and a1(0.9) each differ from BOTH a1(0.5) and a1(1.0), and
  a1(0.5) < a1(0.75) < a1(0.9) < a1(1.0) in mean.
- **DICHOTOMY (any cancellation breaks the clock)** -- a1(0.75) and a1(0.9) do NOT differ from
  a1(0.5), or lie below it, while a1(1.0) differs from all three.
- Anything else: reported as it falls, no mechanism sentence.
Secondary (reported, no verdict): exchange rate k(p) = the fewest index layers in {1, 2, 3} whose
mean accuracy is within 0.01 of the path 1-layer mean at that p (">3" if none); G at 2 and 3 layers;
path 1-layer vs floor at each p; run classes (SOLVED / STALLED / DESCENDING, `classify_run`,
SOLVED = final-5% loss < 0.05); r(final loss, accuracy).
Void: any of the 128 runs missing; md5 guard trips; path 1-layer below its floor at any p.
Scope: 1D ring of 32, T=128, d 128, 2 heads, one recipe, 300-epoch budget.
Cost: ~3-9 s/epoch under the current GPU load; ~40 slot-hours. Runs one job per GPU beside the
train_variant batches.
