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

## Amendment 1 results (2026-09-19) -- the reruns at init 0.1

**G1b Dyck-2 (F1, 8 seeds).** Against the inert twin at L128 D12: zero-init **-0.046** (MDE 0.043,
DETECTABLE, 1/8), init 0.1 **-0.047** (MDE 0.066, unmeasured, 2/8). Same magnitude, same sign, so the
earlier Dyck verdict was NOT an artefact of the bad prior.

| arm | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| MapPoPE-1L baseline | 0.988 | 0.923 | 0.976 | 0.927 |
| T3 zero-init | 0.986 | 0.915 | 0.938 | 0.900 |
| T3 init 0.1 | 0.982 | 0.903 | 0.945 | 0.899 |
| T3 inert twin | 0.993 | 0.952 | 0.977 | 0.946 |

**G2b torus (revisit accuracy, 8 seeds).** Forcing the phase makes it WORSE, monotonically with
length: at l=2048 MapPoPE-Flat 0.963, inert twin 0.963, T3 zero-init 0.942, **T3 init 0.1 0.911**,
with the seed spread growing to +/-0.049 against the baseline's +/-0.006.

**So the boundary is a double dissociation, not a one-sided null.** The same intervention at the same
strength: on a clock accumulator (Bach) forcing the phase is the best configuration measured
(0.6162 at 2-4x, against 4.6158 with no phase); on bounded accumulators (Dyck-2, torus) it is
neutral-to-harmful and forcing it harder makes it worse.

**The registered phase-magnitude readout does NOT support the story I expected**, and is worth
recording for that reason:

| task | accumulator | mean abs phase, zero-init | forced init 0.1 |
|---|---|---|---|
| Bach | clock (alpha 1.00) | 0.412 rad | 1.118 rad |
| Dyck-2 | bounded | 0.570 rad | 0.825 rad |
| torus | map (alpha ~0.5) | 0.251 rad | 0.782 rad |

I predicted the model would drive the forced phase back toward zero where it is not needed. It does
not: on Dyck it keeps 0.825 rad -- MORE than Bach's zero-init model keeps -- while performing worse
than its own inert twin. So the freedom is not declined, it is used and mis-used. "The model only
uses the phase where it pays" is refuted as stated; what is true is narrower: the phase HELPS only
where the accumulator leaves its trained range, regardless of how much of it the model takes up.
