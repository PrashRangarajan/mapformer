# DECAY_PREREG -- a decay envelope on PoPE, for the failure the phase fix also repairs

## Motivation

PoPE's kernel has non-negative amplitudes and constant phases, so beyond the range training calibrated
it returns confident WRONG values rather than decaying to nothing (`THEORY_MAPPOPE.md`). Two repairs
already work: bounding the accumulator (T1) and restoring a per-token phase (T3, 4.616 -> 0.616 at
2-4x). A third is standard in the long-context literature and untried here: an envelope that makes the
kernel's influence decay with distance, so unreliable far pairs are suppressed instead of trusted.
xPos (arXiv:2212.10554) does this for RoPE; ALiBi (arXiv:2108.12409) shows a linear-in-distance
additive bias suffices. This also answers the separate question of what could help PoPE ITSELF on
language, where PoPE alone already extrapolates better than RoPE (1.597 vs 2.059 at 2-4x).

## What is added

`scores -= softplus(lambda_h) * distance`, one learnable scalar per head, initialised on ALiBi's
geometric spread (2^-1 .. 2^-8: some heads local, some global -- a single constant would make every
head local). Parameter cost is n_heads per layer, i.e. 48 on this model, so no inert twin is needed.
Two arms, matching the two position mechanisms:

- `PoPE_decay`: index PoPE, distance `|t - s|` (the classical ALiBi/xPos distance).
- `MapPoPE_decay`: path integration + PoPE, distance `|S_t - S_s|` in units of the model's own mean
  absolute step, so lambda means "decay per token-equivalent" whatever rate the model learned.

Bach Chorales, 512-token training crops, 5 seeds, everything else as `JSBLEN_PREREG.md`. Baselines
already measured: PoPE 0.5403 / 0.6749 / 1.5973; MapPoPE 0.5235 / 1.9536 / 4.6158; MapPoPE with the
per-token phase 0.5182 / 0.5629 / 0.6162; MapWM 0.5382 / 0.7835 / 1.3969.

## Registered verdicts

- **D1** `MapPoPE_decay` at 1024-2048 is far below MapPoPE's 4.616. The envelope should remove the
  collapse for a different reason than T3 does -- not by letting content correct the kernel, but by
  refusing to trust it far away.
- **D2** `PoPE_decay` - `PoPE` at 1024-2048: predicted negative (better). This is the "does another
  positional encoding help PoPE on language" question, on the index row where PoPE lives.
- **D3** In-distribution (0-512) cost: predicted small for both. A decay envelope trades long-range
  reach for local sharpness, so a LARGE in-distribution gain would be as suspicious as a large loss --
  it would mean the task never needed long range and the extrapolation numbers are about something else.
- **D4** Reported without a prediction: `MapPoPE_decay` against `MapPoPE + per-token phase` (0.6162).
  If the envelope matches or beats it, the cheaper repair wins -- 48 parameters against 786k.
- **D5** The learned lambda per head after training, reported for both arms. If the model drives every
  head's lambda toward zero the envelope was not used and D1/D2 are about something else.

**Falsification of the account's scope**: the envelope suppresses far pairs regardless of WHY the
kernel is wrong, so a large D1 gain does not by itself support the out-of-range story. What would
weaken that story is D2 failing while D1 succeeds -- the same envelope helping only where the
accumulator is learned would say the problem is the accumulator, not the kernel's reliability.
