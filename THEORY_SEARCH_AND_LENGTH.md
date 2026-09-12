# Two theories: why half the per-token rewinds are missing, and what the OOD-length axis is

Written 2026-09-12, before either test was run. Both are stated so they can fail, and both have a
first test that is EVAL-ONLY on checkpoints already in hand.

---

# T1. The per-token search is a Diophantine approximation problem, and frequency pruning is how the model makes it solvable

## The setup, exactly

For a query token `q_k` the model may choose `z_k = W_in e(q_k) ∈ R^r` (r = 4) freely, because
`e(q_k)` is a free embedding row. That gives `Delta(q_k) = W_out z_k ∈ R^{H·n_b}` (64 blocks).
Writing `b_i` for everything else the accumulator picks up between the answer and the query --
the intervening symbols, the filler, the query's own MASK -- the kernel at the answer is

    kappa = sum_i a_i cos( omega_i * ( b_i + (W_out z_k)_i ) ),    a_i = |q0_i||k0_i| >= 0

and a rewind is any `z_k` making every live term near its maximum:

    (W_out z_k)_i  ≡  -b_i   (mod 2*pi/omega_i)   for every block i with a_i > 0.

**That is up to 64 simultaneous congruences, with different moduli, in 4 unknowns.** The linear
solution `z_k = -(k-1) z_sym` satisfies them all at once only in the idealised construction where
symbols share a step and filler contributes nothing. In a trained model neither holds, so what
the optimiser is actually doing is SIMULTANEOUS DIOPHANTINE APPROXIMATION in a rank-r subspace.

## The claim

**Three things make the system solvable, and the model uses all three:**

1. **Pruning.** Only blocks with `a_i > 0` constrain anything. Measured (`probe_phase_spread`):
   trained single-`p0` EM leaves **39 of 64 blocks dead**, amplitudes down to 5e-86. That is not
   waste -- it removes 39 congruences. **Frequency pruning is the search strategy.**
2. **Wrapping.** Each surviving congruence is mod `2*pi/omega_i`, and `omega` is geometric over
   ~64x, so low-frequency blocks are nearly unconstrained for small shifts while high-frequency
   ones are nearly free to wrap. This is why every rewind found from scratch is WRAPPED and none
   is linear (0/8 in the fixed-k batch, linear ratios +0.3..+1.4 against a target of -63).
3. **Slack.** Exact zero phase error is not required -- the answer only has to beat its
   competitors, so a weighted error budget suffices.

## Why only some tokens

Each `q_k` faces its own instance, with its own `b` (dominated by `(k-1)*s`), and solves it
alone: `z_k` is private to that token. Two things then decide a token's fate, and the theory says
which one dominates:

- **Existence.** For some k there may be no good `z` in the rank-4 subspace at all -- the
  congruences are unsatisfiable within the achievable slack.
- **Search.** A good `z` exists but is not found, because the token's gradient signal is
  1/m of the query stream (SPREAD2: exposure per token is the currency, +0.665 at fixed budget,
  +0.045 at matched exposure) and the landscape roughens as the kernel sharpens (S2: 0-2 barriers
  at init, ~15 at k=60-64 in a trained model).

**T1 predicts SEARCH dominates**, on the strength of SPREAD2: if the failures were existence, more
exposure per token could not fix them, and it does.

## The decisive test (eval-only, on the 24 stored P0 checkpoints)

Compute, for every token k, the BEST ACHIEVABLE kernel value in the model's own rank-4 subspace:

    Q_k = max over z in R^4 of  sum_i a_i cos(omega_i (b_i + (W_out z)_i)) / sum_i a_i

by multi-start optimisation in 4 dimensions -- cheap, and it uses the model's own trained
`omega`, `W_out` and amplitudes. Then split tokens by whether the model solved them.

- **If failed tokens have HIGH `Q_k`** (a good rewind exists and was not found) -> the deficit is
  SEARCH, per token, as SPREAD2 implies. Predicted.
- **If failed tokens have LOW `Q_k`** -> those tokens are unsolvable in this subspace and the
  deficit is partly REPRESENTATIONAL at the token level, which would be new and would put a floor
  under any exposure-based fix.

This is rule 29 (existence before mechanism) applied one token at a time -- the same move that
killed Thm 3's corollary, now at finer grain.

## Further predictions, in order of cost

- **P1a (free).** Across seeds, the fraction of tokens solved correlates POSITIVELY with the
  fraction of DEAD blocks. Fewer live congruences, easier system.
- **P1b (free).** `EMPair` -- which does not need rewinds at all -- should prune LESS than P0.
- **P1c (free).** Solved k should cluster where the required shift `(k-1)*s` is near a common
  near-period of the live blocks; failed k between them.
- **P1d (one batch).** Force blocks live (penalise amplitude variance, or fix |p0_i| equal) and
  the solved fraction should FALL. This is the causal test of "pruning is the strategy".

## What T1 already explains

The per-token, all-or-nothing pattern; wrapped rather than linear solutions; exposure as the
currency; why one shared k is easy (one instance, all the queries); why a curriculum helps only
the tokens it introduces early; and why the dead blocks are there at all -- which was recorded as
an incidental oddity and is, on this account, the mechanism.

---

# T2. The OOD-length axis is phase-cell exhaustion

## The puzzle

Five mechanisms in this project help specifically at OOD length -- rank r=4, the InEKF, the
forget gate, PoPE, and now EM over WM -- and nothing explains the axis. `alpha` (the accumulator
growth exponent) covers two of them and the imported critical-dimension account was refuted
(ablating low-frequency channels costs MORE at length, not less).

## The claim

**A position code resolves a finite number of distinguishable displacements. Failure at length is
a pigeonhole collision: the number of candidate keys outgrows the number of resolvable phase
cells, and some distractor lands at the same phase as the answer.**

Two quantities, both computable from a trained model:

- **Resolution** `delta`: the smallest displacement that decorrelates the kernel -- the
  autocorrelation width of `kappa(d) = sum_i a_i cos(omega_i d_i + phi_i)`, set by the
  amplitude-weighted high-frequency edge of the live spectrum.
- **Spread** `R(T)`: the range the accumulator actually covers at sequence length T, which grows
  as `T^alpha` (measured: 0.52 signed, 0.94 monotone).

Resolvable cells `N(T) ~ R(T)/delta`; competing keys grow like T. **Accuracy should fall when the
key count overtakes the cell count**, giving a breakdown length

    T* solves  T*^(1 - alpha)  ~  c / delta

## Why this unifies the five

- **rank r=4**: a better-conditioned action basis spreads displacements over more of the space
  (opposition 0.09 vs 0.50, |cos(N,E)| 0.18 vs 0.78) -> larger effective R, more cells.
- **sign / alpha**: cancellation changes what the accumulator measures and hence R(T) -- the two
  cases alpha covers.
- **InEKF**: its per-token gating zeroes `Delta` on non-action tokens, so fewer distinct
  accumulator values are generated per unit length -- fewer keys competing for the same cells.
  This is the account that survives the refutation of "the wrap bounds the accumulator"
  (measured false: range 285.6 vs 283.9).
- **PoPE**: changes the amplitude spectrum, hence `delta` directly.
- **EM over WM at length**: EM's kernel is one shared, coherent function with `kappa(0)` maximal;
  WM's is per-pair, so its effective resolution varies pair by pair and its worst pairs set the
  collision rate.

## The decisive test (eval-only, on the paper-task rerun checkpoints, logs kept)

1. **Error structure.** At l=1024 and l=2048, are wrong retrievals concentrated on keys whose
   displacement is a near-alias of the answer's (`kappa` within the margin), rather than diffuse?
   Predicted: yes, and increasingly so with length.
2. **One number should predict the curve.** Compute `delta` and `R(T)` per checkpoint; the
   predicted collision rate should track measured accuracy ACROSS ARMS (Vanilla, VanillaEM_P0,
   MapPoPE-Flat) and lengths (128 -> 2048), with no free parameters beyond one global margin.

**Falsifier**: if errors are diffuse rather than alias-structured, or if the predicted collision
rate does not order the arms correctly, T2 is wrong and the length axis remains open.

## What T2 would cost the existing account

If T2 holds, `EM_WM_THEORY.md` 2a's third objection -- "the axis is LENGTH, not
offset-fixedness, and length is unexplained" -- stops being an objection and becomes a mechanism:
EM's advantage at length would be a resolution advantage, measurable from its weights, and the
fixed/varying-offset framing would be the wrong cut for that cell.
