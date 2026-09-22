# Pre-registration: PoPE's component ablation, and the extrapolation test their Table 5 cannot do

Written before any run. Arms verified before launch: `PoPE-Full` reproduces
`PoPE-Flat` to **maxdiff 0.00e+00**; `NoSigma`/`ReLU` differ at init; `NoDelta`
is identical at init (delta is zero-initialised) and diverges once full PoPE
learns a non-zero delta -- confirmed, maxdiff 6.2e-02 with delta perturbed, and
its delta is frozen (`requires_grad=False`, value 0) in every layer.

## What the paper reports (Table 5, OpenWebText val ppl, 124M / 253M)

| arm | 124M | 253M |
|---|---|---|
| PoPE without sigma() | 21.57 | 18.93 |
| PoPE with ReLU for sigma() | 21.55 | 18.90 |
| PoPE without delta | 21.42 | 18.57 |
| **Full PoPE** | **21.33** | **18.55** |

Never checked in this repo. That table is **in distribution**.

## The account under test, which their table cannot address

PoPE's magnitudes are non-negative, so for every offset, seen or unseen:

    |score(D)| = |sum_c mu_q mu_k cos(w_c D - d_c)| <= sum_c mu_q mu_k = score(0)

RoPE's signed Q.K has no such bound. Measured on the trained code checkpoints,
PoPE's kernel peaks at 0.3528 beyond the training context against 0.9954 inside
it, while a signed-weight kernel EXCEEDS its in-training maximum at unseen
distance. This is the standing explanation for why PoPE-encoding arms degrade
gracefully past their context and RoPE-encoding arms blow up.

**ReLU is the discriminating arm.** It is equally non-negative, hence equally
bounded, yet in distribution the paper shows it behaving like NoSigma. So:

- **A1 (primary).** At 2048 (4x the training context), **ReLU extrapolates like
  Full PoPE** -- i.e. `ReLU - Full` at 1024-2048 is within its MDE -- **while
  being detectably worse in distribution**. That is the bound account's
  signature and nothing else predicts it.
- **A2.** `NoSigma` (signed magnitudes, no bound) loses the extrapolation
  advantage: detectably worse than Full at 1024-2048, and by more than its
  in-distribution deficit.
- **A3, replication.** In distribution the ordering reproduces theirs:
  Full <= NoDelta < ReLU ~ NoSigma.

## Falsifiers

- **F1, kills the bound account.** ReLU extrapolates detectably WORSE than Full
  at 1024-2048. Then non-negativity is not what buys extrapolation, and the
  account that has been carrying the code line dies.
- **F2.** NoSigma extrapolates as well as Full. Then the bound is irrelevant in
  both directions and the ordering is about something else entirely.
- **F3.** No arm separates from any other at 1024-2048 beyond MDE. Then this
  corpus cannot resolve the question and the verdict is "unmeasured" (rule 11),
  not a null.

## Design

Code corpus, **seq 512**, then evaluated at **2048** with the committed
`eval_code_long.py` -- the same data and readout the original effect was
measured on. 4 arms x 3 seeds, all in ONE batch (`PoPE-Full` is retrained here,
not taken from `runs/code`, per rule 3). 36k iters, bs 16, lr 2e-4, dim 512,
9 layers. Readouts: val bpc by position bucket (0-512 / 512-1024 / 1024-2048)
and closer accuracy by bracket distance, with the no-stack floor beside it.

MDE = 2.8*sd/sqrt(3) per paired contrast; anything not clearing it is
"unmeasured" with the MDE stated. Void if any arm is unconverged or if the
`PoPE-Full` control fails to land near the stored `PoPE-Flat` value (0.9131).
