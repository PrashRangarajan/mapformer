# Pre-registration: does the OOD-length benefit localise to under-trained channels?

Written 2026-09-06 before running anything. Eval-only on existing checkpoints.

## The hypothesis, imported from the long-context literature

Channel $c$ carries phase `theta_c = omega_c * S_t`, where `S_t = cumsum(Delta)`.
The long-context line (LongRoPE2's critical dimension) argues that a channel is
trained through its whole phase range only if it completes at least one full cycle
within the training distribution; channels that do not are read at unseen phases
when the input grows, and that is the mechanism of length-extrapolation failure.

Transposed to a content-dependent phase, the coordinate is the accumulator, not the
index. Define the **cycle count** of channel `c` at length `T`:

    n_cycles(c, T) = omega_c * range_T(S) / (2*pi)

where `range_T(S)` is the spread of the accumulator over trajectories of length `T`.
A channel is **under-trained** if `n_cycles(c, 128) < 1` at the training length.

If the account is right, the OOD-length signature this project keeps reporting --
rank, the InEKF, the forget gate and PoPE all help at extrapolated length and only
there -- is those channels going out of range.

## Predictions, and what refutes each

**P1 (necessary condition, descriptive).** Some channels must be under-trained at
`T=128`. If `n_cycles(c,128) >= 1` for every `c`, the mechanism cannot operate here
at all and P2/P3 are void.

**P2 (the localisation test, causal).** Ablate a channel's phase at eval time by
setting `theta_c = 0` (that channel then contributes no rotation, i.e. NoPE on that
channel). Ablate the `k` lowest-frequency channels, the `k` highest-frequency
channels, and `k` random channels, at `T=128` and `T=1024`, sweeping `k`.

- **Predicted:** ablating LOW-frequency channels costs *less at T=1024 than at
  T=128*, because at OOD length they are already being read out of distribution.
  The strong form is that it *helps* at T=1024 -- removing a channel that is
  contributing noise should raise accuracy.
- **Refuted if:** ablating low-frequency channels costs the same or more at OOD than
  at training length, or costs the same as ablating high-frequency channels. Either
  means the damage is not localised in frequency and the account is wrong.
- **Uninformative if:** all ablations are inside the noise floor at every `k` --
  64 channels are redundant enough that removing a few may do nothing. Report the
  `k` sweep, not a single `k`.

**P3 (does the account explain the ARMS?).** The arms differ in OOD accuracy by
known amounts. Compute the fraction of under-trained channels and the growth of
`range(S)` from `T=128` to `T=1024` per arm.

- **Predicted:** arms that hold up at OOD (`Signed_r4`, `r=4`) show a smaller
  accumulator growth or fewer out-of-range channels than arms that collapse
  (`Abs_r4`, `r=2`).
- **Refuted if:** the ordering does not track OOD accuracy, or is flat.

## Arms and analysis

`Signed_r4` and `Abs_r4` from the sign batch (12 seeds each) -- a pair with a known,
large OOD gap and identical parameters. `Vanilla` (r=2) and `Vanilla_r4` from the
rank sweep if the checkpoints load, for the second known OOD contrast.

Paired per seed; MDE = `2.8*sd/sqrt(n)`; anything inside its MDE is reported as
unmeasured, not as a null. P2 is the primary; P1 gates it; P3 is corroborative and
correlational by construction.
