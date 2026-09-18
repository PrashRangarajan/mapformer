# T3_PREREG -- give PoPE's phase a per-token component (THEORY_MAPPOPE.md)

## What is added

`model_pope_t3.py`: `delta` becomes `delta_c + d^q_c(x_t)` on queries and `delta_c + d^k_c(x_s)` on
keys, from two zero-initialised linear heads, so training starts as exactly PoPE. This restores the
PAIRWISE phase freedom that rotating Q and K gives MapWM for free, while keeping the path phase.
Cost: +786k parameters on the JSB model (4.79M -> 5.57M), so an INERT TWIN is run alongside --
identical parameters, phase heads present and gated to zero. Verified before launch: with weights
copied the twin is bitwise identical to plain MapPoPE (max |diff| 0.00e+00), its phase heads receive
exactly zero gradient, and T3's receive non-zero gradient.

## Why this is the discriminating test

T1 rescued MapPoPE by shrinking the accumulator, but it also cost 0.13 NLL in distribution, and a
sceptic can read that as "a weaker positional signal makes the model lean on content" rather than as
the account's "the kernel argument left its calibrated range". T3 separates them, because it changes
NOTHING about the accumulator -- same clock, same range, same alpha -- and only restores the
compensating degree of freedom. The account therefore predicts a DOUBLE-SIDED result:

1. extrapolation recovers WITHOUT the in-distribution cost centring imposed, and
2. PoPE's pure-indexing advantage is partly given back, because a content-dependent phase is exactly
   the what/where re-entanglement PoPE exists to prevent.

A one-sided outcome -- extrapolation fixed at no cost anywhere -- would mean the account is too
simple and the pairwise phase is a free lunch, which nothing in this project's experience suggests.

## Part A: Bach Chorales, training context 512 (5 seeds)

Arms: `MapPoPE_T3`, `MapPoPE_T3inert`. Baselines already run: `MapPoPE_r2` (4.616 at 1024-2048,
0.5235 in distribution) and the centred arm (0.911 / 0.656).

- **A1** T3 - MapPoPE at 1024-2048: predicted large and negative.
- **A2** T3 - MapPoPE at 0-512: predicted NOT detectably worse. Centring cost +0.133 here; if T3
  costs about the same, it is behaving like a weaker position signal and does not separate the
  accounts.
- **A3** inert twin - MapPoPE at every bucket: predicted unmeasured. If the twin alone improves
  extrapolation, the effect is parameters, not phase, and A1 means nothing.

## Part B: Indirect Indexing at 100,000 iterations (8 seeds), queued after Part A

Arms: `MapPoPE_T3`, `MapPoPE_T3inert`. Baselines already run at this budget: MapPoPE 5/8 solved,
PoPE 1/8, MapWM 0/8, RoPE 0/8.

- **B1** T3's solve rate: predicted LOWER than MapPoPE's 5/8, moving toward PoPE's 1/8. This is the
  predicted cost, and it is the half of the prediction that can embarrass the account.
- **B2** inert twin's solve rate: predicted ~5/8. A drop here would mean the extra parameters make
  the search harder, and B1 would be uninterpretable.
- Reported: lift-off step, since a slower solve at the same rate is a different result from a lower rate.

## Falsification summary

- A1 fails -> the pairwise-phase account of the collapse is wrong.
- A1 passes and A2 fails (T3 costs as much in distribution as centring) -> T3 is not distinguishable
  from a weaker position signal; the account is not separated from the sceptical reading.
- A1 and A2 pass and B1 shows no cost -> the account is too simple; a free lunch needs an explanation
  the account does not have.
