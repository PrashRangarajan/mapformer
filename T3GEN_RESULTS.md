# T3 generality and the forced-phase test -- the account's boundary holds, my post-hoc reading dies

Pre-registration: `T3GEN_PREREG.md`. Three batches, all with the parameter-matched inert twin where
applicable.

## G1 -- Dyck-2 (8 seeds). F1, higher is better. Prediction: LITTLE effect

| arm | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| MapPoPE-1L (baseline) | 0.988 | 0.923 | 0.976 | 0.927 |
| MapPoPE_T3 | 0.986 | 0.915 | 0.938 | 0.900 |
| MapPoPE_T3 inert twin | 0.993 | 0.952 | 0.977 | 0.946 |

Against its own inert twin -- the controlled comparison -- T3 is **-0.046 at L128 D12** (DETECTABLE,
1/8 seeds better) and negative at every other cell. **The registered prediction holds in direction:
no gain where the accumulator is bounded.** It is slightly worse than that -- the phase carries a
small cost when there is nothing for it to absorb. Dyck's increments cancel (opens push, closes pop),
so `S_t - S_s` never leaves its trained range, and the freedom that rescued music buys nothing here.

## G2 -- torus paper task (8 seeds). Revisit accuracy. Prediction: LITTLE effect

| arm | IID l=128 | OOD-d | OOD-s l=512 | ext l=1024 | ext l=2048 |
|---|---|---|---|---|---|
| MapPoPE-Flat | 1.000 | 0.992 | 0.990 | 0.979 | 0.963 |
| MapPoPE_T3 | 0.999 | 0.992 | 0.978 +/- 0.024 | 0.961 +/- 0.042 | 0.942 +/- 0.046 |
| MapPoPE_T3 inert twin | 1.000 | 0.993 | 0.991 | 0.979 | 0.962 |

Same verdict, second bounded accumulator: no gain, a slight cost at long evaluation, and a seed
spread 8-9x the baseline's (0.046 vs 0.005) -- the phase mostly adds variance here. The inert twin
tracks the baseline to three decimals at every length, so this is the phase, not the parameters.

## G3 -- forced phase on Bach (5 seeds). Test NLL, lower is better

| arm | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|
| MapPoPE (no per-token phase) | 0.5235 | 1.9536 | 4.6158 |
| T3, zero-initialised | 0.5594 | 0.6411 | 0.7325 |
| T3, forced init 0.02 | 0.5538 | 0.6186 | 0.7123 |
| **T3, forced init 0.1** | **0.5182** | **0.5629** | **0.6162** |

**My post-hoc reading is refuted, and in the opposite direction.** I proposed that the phase is
optional freedom which a model declines when it needs positional purity, and predicted that forcing
it would COST something. Forcing it **helps, monotonically in the initialisation scale**: at 0.1 it is
better than zero-initialised T3 in distribution (-0.041, 5/5, detectable) AND at 2-4x (-0.116, 5/5,
detectable). The zero start is simply a bad prior -- the model under-uses the phase from there.

Consequence: **forced T3 dominates plain MapPoPE at every bucket**, including in distribution
(0.5182 vs 0.5235), so T3's one remaining cost disappears with a better initialisation. On this task
it also beats MapWM everywhere (MapWM: 0.538 / 0.784 / 1.397).

## What survives

**The account's boundary is now tested on three tasks and holds.** A per-token phase pays exactly
where the accumulator leaves the range training calibrated -- music, a clock with alpha = 1.00 -- and
does nothing, or slightly hurts, where the accumulator is bounded: Dyck-2 (cancelling increments) and
the torus (a map, alpha ~0.5). That is the prediction registered before these runs, and it is the
kind of prediction that could have failed cleanly.

**Two of my explanations have now died in this line**: the trade-off corollary (T3 costs indexing
purity -- it does not) and the optional-freedom reading (forcing the phase costs something -- it
helps). What is left is narrower and better supported than what I started with: the collapse is an
out-of-range accumulator meeting a kernel that cannot compensate, both halves have interventional
support, and the compensating mechanism is worth having only when the first half is present.

**Open, not run**: whether forced-phase T3 helps on Dyck and the torus too (G1/G2 used the zero
initialisation, which G3 shows is the wrong one), and whether the gain survives at a lower parameter
cost than the 786k these heads add.
