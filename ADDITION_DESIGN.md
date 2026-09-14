# Multi-digit addition: can a learned path integrator discover position coupling?

Design note, written 2026-09-14, before any code or GPU. Gate and pilot come first; the
pre-registration is written only after the pilot shows the task is learnable and off both the
floor and the ceiling.

## What is already known (read first-hand; `papers/txt/position_coupling.txt`,
`papers/txt/zhou_length_gen_not_robust.txt`)

- **Format matters most.** Reversed answer plus zero-padded operands is the standard format
  (Lee et al. 2024). Without zero-padding, even position coupling scores 0.0% at 200 digits (Cho et
  al. 2024, ablation).
- **Index codes do not length-generalise.**
  - Zhou et al. 2024 reach 2.5x (train 40 digits, test 100), but only with FIRE, randomised
    positions, reversed format and index hints, and robustness depends on the seed.
  - NoPE and random-start APE generalise only to about 34 digits after training on 30 (Cho et al.).
- **Hand-assigned position coupling generalises.** Cho et al. (NeurIPS 2024, arXiv:2405.20671) give
  the same position ID to digits of the same significance. Training on 1-30 digits then reaches
  95.65% at 200 digits with 1 layer and 4 heads.
  - They prove a 1-layer Transformer with coupled positions can add exponentially long numbers.
  - They also prove no 1-layer Transformer without positional information can.
- **Abacus embeddings** (McLeish et al. 2024, arXiv:2405.17399) give each digit its index within its
  own number and generalise 6x.
- **The open slot, in Cho et al.'s own words:** "uncovering hidden structures and autonomously
  creating appropriate couplings (without manually designing them)" is left for future work
  (Limitations).

## The question

MapFormer's position is a running sum of learned per-token increments. Can it LEARN a position
coupling instead of being handed one?

## What each architecture can represent (algebra; to be checked by construction before any claim)

Format: `$ a_{n-1}..a_0 + b_{n-1}..b_0 = s_0..s_n $`. Operands are MSB first and zero-padded to n;
the sum is reversed. Attention depends on S_t - S_s. Coupling needs the sum digit of significance j
to sit at a FIXED offset from a_j and from b_j, for every j and every n.

- **Shared digit tokens (F1, the literature's format).** Every digit gets the same increment delta
  within a head, and '+' and '=' get constants c and d. Then S(s_j) - S(a_j) = (n + 2j + 1) delta
  + c + d, which depends on j and n unless delta = 0.
  - So neither index RoPE (delta = 1) nor MapFormer can express coupling in F1. That includes
    MapFormer's per-head increments, and wrapping modulo each block's period does not help beyond
    degenerate frequencies.
  - F1 is therefore the control, where a representational limit is predicted for all learned codes.
- **Role-tagged digit tokens (F2).** The three groups (first operand, second operand, sum) use
  disjoint copies of the ten digits, so delta can differ by role. MapFormer's increments are
  per head.
  - A head with delta_a = +1, delta_b = 0, delta_s = -1 puts s_j at a fixed offset from a_j.
  - A second head with delta_a = 0, delta_b = +1, delta_s = -1 does the same for b_j.
  - So a 1-layer, 2-head MapFormer can express exact, length-invariant coupling. It needs a
    NEGATIVE increment (delta_s = -delta_a), which ties this directly to the sign result.
  - Index RoPE cannot, because every token's increment is 1. A monotone MapFormer (|delta|) cannot
    either, except approximately through wrapping.
- **Carry.** The carry needs s_j to also see a_{j-1} and b_{j-1}, which is one constant offset
  further. Any code that expresses the coupling can express this too.

"Can represent" is not "will learn": MapFormer-EM's recency rewind was representable and hard to
find. A construction (install the coupling increments, freeze them, train the rest) is the planned
existence check.

## Arms for the pilot (1 seed each, both formats)

| arm | position | why |
|---|---|---|
| `RoPE` | index | standard baseline |
| `NoPE` | none | literature baseline for length generalisation |
| `Vanilla_r4` | path-integrated, signed | the question |
| `Abs_r4` | path-integrated, monotone | the sign control: predicted unable to couple |
| `CoupledRoPE` | hand-assigned coupled IDs (Cho et al.) | upper bound / positive control |

## Protocol

- **Sampling.** Balanced over digit counts (Cho et al.): draw D uniformly from 1..Dmax, then the
  operand uniformly among D-digit numbers.
- **Lengths.** Train Dmax = 16 in the pilot; evaluate exact-match accuracy at 8, 16, 24, 32, 48
  and 64 digits.
- **Loss** on sum tokens only.
- **Model.** Pilot uses 1 and 2 layers, 4 heads, d = 256.
- **Gates, before any GPU:**
  - n-gram predictors of orders 1-5 on the sum stream alone must sit near the per-digit chance of
    0.1;
  - a "copy the aligned digits without carry" predictor is recorded as a reference, not a shortcut;
  - exact-match chance is reported per length.
- **Pilot pass condition.** Some arm is off both the floor and the ceiling at training length, so
  the comparison can move. The pre-registration follows only then.
