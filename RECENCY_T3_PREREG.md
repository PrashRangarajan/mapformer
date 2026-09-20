# RECENCY_T3_PREREG -- the per-token phase where the accumulator is a CLOCK inside the torus family

## Why this task

`T3GEN_RESULTS.md` established a double dissociation: a per-token phase added to PoPE pays where the
accumulator leaves its trained range (Bach, alpha = 1.00: 4.616 -> 0.616 at 2-4x) and is neutral to
harmful where it is bounded (Dyck-2 -0.047 against its inert twin; torus 0.911 vs 0.963 at l=2048).
Every positive so far is on borrowed data. Recency is the case that tests the rule inside this
project's own line: `RECENCY_RESULTS.md` measured the SAME architecture adopting alpha = 0.591 on the
torus and **alpha = 0.967 on recency** (t ~ 42), with constrained arms pinned at ~1.0 on both as the
control. So recency is a clock living in the torus family, and the rule predicts the phase pays here
and not on the torus -- two tasks, one codebase, opposite predictions.

## Arms and recipe

`run_recency.sh` verbatim: k_max 64, T = 1024, 300 epochs, 48 batches of 16, lr 1e-3, cosine,
1 layer, 2 heads, d_model 128, grid_size 64, evaluated at T = 1024 (training length) and 2048.
8 seeds, one batch, three arms:

- `MapPoPE-Flat` -- the baseline. Note it has no recency numbers yet: the table quoted as a recency
  baseline in `MAPPOPE_R4_RESULTS.md` is in fact that file's TORUS table, so the baseline is run here.
- `MapPoPE_T3pi01` -- per-token phase, forced initialisation 0.1 (G3 showed the zero start is a bad prior).
- `MapPoPE_T3inert` -- parameter-matched inert twin, phase gated to zero.

## Registered verdicts

- **R1 (the prediction)** T3pi01 - inert twin at **T = 2048**: positive and detectable. This is the
  out-of-range condition the account says the phase is for, on a clock accumulator.
- **R2** T3pi01 - inert twin at **T = 1024** (training length): predicted small. A large gain at
  training length would mean the phase is buying something other than range compensation.
- **R3** inert twin - MapPoPE-Flat: predicted unmeasured at both lengths. If the parameters alone
  help, R1 is uninterpretable.
- **R4 (the cross-task contrast that is the actual claim)** the sign of R1 against the torus result
  for the same intervention (-0.052 at l=2048). Same architecture, same phase, same initialisation:
  **positive here and negative there is the claim; the same sign in both refutes the rule.**
- Reported: alpha per arm as the manipulation check (recency should be ~0.97, not ~0.59), the learned
  phase magnitude, and rule 9's r(final loss, accuracy) since recency accuracy has tracked convergence
  in earlier batches.

**Falsification**: if the phase does not help at T = 2048 on a measured clock accumulator, the rule
that survived three tasks does not generalise beyond the two datasets it was built on, and the honest
statement becomes that the fix is Bach-specific.
