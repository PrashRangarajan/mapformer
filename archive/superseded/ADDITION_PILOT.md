# Addition pilot: learnability and range (ADDITION_DESIGN.md)

One seed per cell. This is a pilot: it establishes whether the task is learnable and off both the floor
and the ceiling. It is **not** a result. 5 arms x 2 formats x {1, 2} layers, d=256, 4 heads, trained on
1-16-digit operands for 100 epochs x 100 batches x 256 (about 2.6M problems, about 1/20 of Cho et al.'s
budget), cosine, lr 1e-3. Held-out problems, both operands with exactly n digits, 512 per length.
Gates: `ADDITION_GATES.md` (pass).

## Exact-match accuracy by operand length (trained up to 16 digits)

| layers | format | arm | final loss | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | role | `Vanilla_r4` (signed) | 0.002 | 1.00 | 1.00 | **0.71** | 0.01 | 0.00 | 0.00 |
| 1 | role | `Abs_r4` (monotone) | 0.410 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | role | `RoPE` | 1.253 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | role | `NoPE` | 1.866 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | role | `CoupledRoPE` (oracle) | 0.001 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | shared | `Vanilla_r4` | 1.481 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | shared | `Abs_r4` | 1.423 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | shared | `RoPE` | 1.654 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | shared | `NoPE` | 2.145 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | shared | `CoupledRoPE` | 0.002 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | role | `Vanilla_r4` | 0.000 | 1.00 | 1.00 | **0.61** | **0.12** | 0.00 | 0.00 |
| 2 | role | `Abs_r4` | 0.004 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | role | `RoPE` | 0.018 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | role | `NoPE` | 0.652 | 0.92 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | role | `CoupledRoPE` | 0.000 | 1.00 | 1.00 | 0.32 | 0.00 | 0.00 | 0.00 |
| 2 | shared | `Vanilla_r4` | 0.012 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | shared | `Abs_r4` | 0.009 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | shared | `RoPE` | 0.011 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | shared | `NoPE` | 1.437 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | shared | `CoupledRoPE` | 0.000 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Learned increments, one layer, role format (eval-only probe; description, not a test)

The mean increment per token role (first operand a, second operand b, sum s) per head, compared across
the 32 frequency blocks with each block weighted by |omega|:

| arm | head | cos(s, a) | cos(s, b) | cos(a, b) | spread over digit values (a) |
|---|---|---|---|---|---|
| `Vanilla_r4` | 0 / 1 / 2 / 3 | -0.72 / -0.78 / -0.64 / -0.62 | -0.77 / -0.70 / -0.78 / -0.72 | +0.11 / +0.09 / +0.02 / -0.11 | 0.05-0.06 |
| `Abs_r4` | 0 / 1 / 2 / 3 | +0.77 / +0.73 / +0.68 / +0.56 | +0.75 / +0.52 / +0.49 / +0.52 | +0.81 / +0.61 / +0.68 / +0.66 | 0.28-0.41 |

In every head, the signed model's sum-digit increment points AGAINST both operand increments. The two
operands use roughly orthogonal directions, and the increment depends on a digit's role, not its value
(spread 0.05). That is the structure position coupling needs: the sum counts back down what each operand
counted up. The monotone model cannot do this and does not.

## What the pilot shows (n=1; nothing here is a claim)

- **The task is learnable and in range.** Several arms solve 16 digits and fail beyond, so length
  generalisation can move.
- **One layer, role format.** Signed MapFormer is the only learned position code that solves the training
  length. RoPE, NoPE and monotone MapFormer do not. It is also the only arm, oracle included, that
  generalises at all (0.71 at 24 digits).
- **One layer, shared format.** No learned code solves even the training length. The coupled oracle
  does. This matches the prediction that shared digit tokens make coupling inexpressible for any
  token-driven code.
- **Two layers.** RoPE and monotone MapFormer also solve 16 digits in both formats, but only role-format
  signed MapFormer (and weakly the oracle) extends past 16.

## Problems to fix before a pre-registered batch

1. **The positive control does not length-generalise.** Our coupled oracle is RoPE over coupled IDs, and
   it scores 0.00 at 24 digits. Cho et al. reach 95.65% at 200 digits with 1 layer, using LEARNED
   ABSOLUTE embeddings over coupled IDs with random starts, trained on 1-30 digits for about 50M problems.
   Either the rotary form, the 20x smaller budget, or the shorter training length is the cause. A faithful
   `CoupledAPE` arm is needed. Until an oracle generalises, "does MapFormer learn coupling" has no
   ceiling to compare against.
2. **Budget and training length.** Everything here is about 1/20 of the literature's budget, trained to 16
   digits. Length generalisation may need both raised.
3. **Seeds.** n=1. The first-seeds overestimate has occurred four times in this project.
