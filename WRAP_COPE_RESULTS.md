# Does a near-wrap start let CoPE learn the ring map? -- results (2026-10-14)

Pre-registration `WRAP_COPE_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-14/wrap_cope_audit.md`). Batch
15:11-17:54 at 622eb74 on both GPUs; ring 32, p_plus 0.5, T = 128, 300 epochs; faithful CoPE (Listing 1); seeds 900-907, n = 8.
Registered output `WRAP_COPE_ANALYSIS.txt`, `WRAP_COPE_VERDICTS.json`. Floor 0.554. Arms differ only in the position table's
initial values.

| arm | position table at init | acc T = 128 | SOLVED | held-out NLL |
|---|---|---|---|---|
| CoPEF | zeros (the paper's) | 0.840 (0.58-1.00) | 1/8 | -- |
| CoPEF_Hi | sinusoid, omega in [0.6, 0.95] pi (an out-and-back past pi) | 0.913 (0.63-1.00) | 2/8 | 0.289 |
| CoPEF_Lo | sinusoid, omega in [0.1, 0.45] pi (below pi) | 0.867 (0.63-1.00) | 1/8 | 0.458 |
| CoPEF_LoS | Lo, amplitude slope-matched to Hi | 0.943 (0.83-1.00) | 0/8 | 0.204 |

## Registered
All UNMEASURED: W1 Hi - Lo +0.046 [95% CI -0.095, +0.180] (p 0.49); W1b Hi - LoS -0.030 (p 0.65); W2 Hi - CoPEF +0.073 (p 0.32);
W3 Lo - CoPEF +0.027 (p 0.70); NLL contrasts W1n / W1bn NO DIFFERENCE DETECTED (p 0.44, 0.70). Seeds are bimodal (0.58-1.00
within an arm), so n = 8 cannot resolve effects of this size.

## Secondaries
- **Every arm's learned table ends with most of its power near pi** -- the period-2 structure of the constructed solution:
  0.40 (zeros init), 0.52 (Hi), 0.55 (Lo), 0.68 (LoS). CoPE finds the periodic table from any start, including zeros.
- Accuracy at 4T and loss AUC favour the sinusoidal starts (LoS 0.891 / 0.565, Hi 0.854 / 0.648, CoPEF 0.742 / 0.854), but 4T
  reads untrained table entries (disclosed in Amendment 1) and neither is registered.
- Dropout re-score changes nothing (<= 0.002).

## What it means
- **The design implication does not transfer to CoPE in this form (unmeasured).** In CoPE the wrap lives in the position
  table, which is an unconstrained parameter per count; training moves it to period 2 from any start. The monotone part is
  the gated count, and what the map needs there is a precise asymmetry between the two moves' gates (17/16 vs 15/16 per pair
  in the construction) -- not a start past pi. So "start near the wrap" is the right lever for a monotone ROTARY step (where
  the step itself carries the wrap: MECHANISM), not obviously for a count read through a free table.
- CoPE's failure on the ring is real but noisy: 1/8 SOLVED here, 0/8 in GENERALITY; the sinusoidal starts trend better
  (0.87-0.94 vs 0.84) without a registered effect.

## Caveats
One task, n = 8 with bimodal seeds, 300 epochs; the table manipulation is our design (the paper initialises at zero).
