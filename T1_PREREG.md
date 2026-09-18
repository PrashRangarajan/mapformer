# T1_PREREG -- does bounding the accumulator rescue MapPoPE? (THEORY_MAPPOPE.md)

## Why this is the test

The account in `THEORY_MAPPOPE.md` says MapPoPE's length-extrapolation collapse is the conjunction of
(i) an accumulator that grows linearly on data whose increments do not cancel, so `S_t - S_s` leaves
the range training calibrated, and (ii) PoPE's kernel having no pairwise phase with which to
compensate, because `pope_delta` is a per-head constant and the amplitudes are non-negative.

(ii) cannot be removed without redesigning PoPE into something else. (i) can: centre the increment.
`ActionToLieAlgebra` is linear, so subtracting the frequency-weighted mean embedding makes
`E[Delta] = 0` exactly, and `S` becomes a mean-zero random walk. That turns the music accumulator
toward the Dyck-2 one -- where increments cancel by construction and MapPoPE is the BEST arm out of
distribution -- while leaving the architecture, the data, the recipe and the relative (difference-only)
structure of the score untouched. Wrapping `S` or squashing it with a tanh would also bound it but
would break additivity, confounding the test.

**This is a falsifier, not a fix**: the account predicts recovery, and the interesting outcome is the
one that kills it.

## Honest limit of the manipulation

Centring makes growth sqrt(T), not constant. At 4x the training context the out-of-range excess
falls from 4x to 2x -- a PARTIAL manipulation. Consequences, registered now:

- a LARGE recovery supports the account;
- NO recovery (MapPoPE still ~4.6 with alpha ~0.5) refutes it;
- a SMALL recovery is ambiguous between "the account is right and the manipulation was too weak" and
  "the account is wrong", and will be reported as ambiguous rather than claimed either way.

## Arms

MapPoPE and MapWM, centred, r=2, base 2048, training context 512, 5 seeds, everything else identical
to `runs/jsb_len512`, which supplies the uncentred baselines.

## Registered verdicts

- **M1 (manipulation check, comes first)** alpha falls from ~1.00 to ~0.5 and range(S) at 2048 falls
  by roughly 2x, on BOTH arms. If alpha does not move, the manipulation failed and nothing else in
  this run is interpretable.
- **P1** MapPoPE centred - MapPoPE uncentred at 1024-2048, paired by seed, MDE 2.8 sd / sqrt(5).
  Predicted: large and negative (better).
- **P2** MapWM centred - MapWM uncentred at the same bucket. Predicted: much smaller than P1, since
  MapWM already compensates through its pairwise phase. **If the two move by similar amounts the
  account is not supported**, by the same logic that killed the rank and omega-base accounts: a knob
  that helps both rows equally is a general extrapolation knob, not an explanation of MapPoPE.
- **P3** In-distribution (0-512) NLL must not be badly hurt for either arm; a large in-distribution
  regression would mean centring damaged the model rather than only its growth rate, which would
  confound P1 and P2.
- **P4** Reported without a prediction: whether centred MapPoPE beats uncentred MapWM (1.397) at
  1024-2048. That is the level at which "the combination finally works" would be a real claim.
