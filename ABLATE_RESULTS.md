# PoPE component ablation: the bound account is DEAD, and their Table 5 does not replicate

Pre-registration `ABLATE_PREREG.md`. Code corpus, 4 arms x 3 seeds in one batch,
trained at 512, evaluated at 2048. All arms converged.

## A3 -- in distribution (their Table 5): DOES NOT REPLICATE

| arm | ours, best val bpc | vs Full | theirs, 124M ppl |
|---|---|---|---|
| PoPE-Full | 0.9135 | -- | 21.33 |
| **PoPE-NoSigma** | **0.9133** | -0.0002 (MDE 0.0051, 2/3) **unmeasured** | 21.57 *(their worst)* |
| PoPE-NoDelta | 0.9148 | +0.0012 (MDE 0.0016) unmeasured | 21.42 |
| **PoPE-ReLU** | **0.9236** | **+0.0101 (MDE 0.0011, 0/3) DETECTABLE** | 21.55 |

They report sigma() as the most load-bearing component. Here **removing softplus
entirely is free**, and **replacing it with ReLU is the only detectable
degradation** -- their ordering inverted on the arm they emphasise. Control:
`PoPE-Full` 0.9135 against the stored `PoPE-Flat` 0.9131.

## A1/A2 -- out of distribution: F2 FIRES, the bound account is dead

| arm | 0-512 | 512-1024 | 1024-2048 | extrapolation penalty |
|---|---|---|---|---|
| PoPE-Full | 0.8763 | 0.7775 | 0.8388 | **-0.0375** |
| PoPE-NoDelta | 0.8771 | 0.7732 | 0.8297 | -0.0474 |
| **PoPE-NoSigma** | 0.9058 | 0.7947 | 0.8824 | **-0.0234** |
| PoPE-ReLU | 0.8892 | 0.7926 | 0.8758 | -0.0134 |
| *RoPE (original batch)* | 0.8755 | 2.6713 | 4.4641 | **+3.5885** |

Contrasts against Full at 1024-2048, all **unmeasured** at n=3:
ReLU +0.0371 (MDE 0.0375, 0/3); NoSigma +0.0436 (MDE 0.0933, 0/3);
NoDelta -0.0091 (MDE 0.0109, 3/3).

**The registered account is refuted.** It said PoPE's non-negative magnitudes
bound the logit by `score(0)` at every offset, and that this is why
PoPE-encoding arms survive past their context while RoPE-encoding arms blow up.
**NoSigma removes the softplus -- signed magnitudes, exactly like RoPE -- and
does not blow up** (-0.0234 against RoPE's +3.5885). Non-negativity is not what
buys extrapolation. F2 as registered.

## What survives: a HYPOTHESIS, not a finding

NoSigma keeps a property the bound account conflated with non-negativity: in
PoPE the **phase is purely positional** (`t*theta_c + delta_c`, delta a per-head
CONSTANT) and content only scales each channel. Dropping softplus makes that
scale signed but leaves the kernel shape **identical for every query-key pair**.
RoPE rotates the content vector, so its phase offset is content-determined and
differs per pair. The load-bearing property therefore looks like a
**content-independent, shared kernel**, not a bounded one.

Consistent with everything measured here, and with the code result that MapWM
(path-integrated RoPE) blows up while MapPoPE does not despite an identical
accumulator. **Untested in its own right.** Clean test: keep softplus and make
the PHASE content-dependent; that should blow up. Seventh prediction in this
line -- the previous six failed, so this is recorded as a hypothesis.

## Caveats

- n=3: nothing separates the variants out of distribution. This establishes
  "none of them break", not an ordering.
- `PoPE-NoSigma_s2` is anomalous (0.9647 at 0-512 against ~0.877 for its
  siblings) and is what inflates that arm's MDE to 0.0933. Plausible instability
  from unbounded magnitudes; not part of the claim.
- Scope: one corpus, 9 layers, 512 training context.
