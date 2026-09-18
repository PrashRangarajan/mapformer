# JSBLEN_PREREG -- length extrapolation on Bach Chorales

Written after a 60-iteration smoke run, before any full run. This tests the axis PoPE's Figure 2
is about (length extrapolation, which they measure on PG-19 after OpenWebText pretraining -- out of
reach here) on a dataset they do use, and it is the axis on which path integration wins elsewhere in
this project (Dyck-2, the torus, MiniGrid).

## Design

Identical to `JSB_PREREG.md` -- same data, same recipe (d256, 8 heads, 6 layers, dropout 0.2, batch
4, lr 6e-4 cosine to 6e-5, 3,000 iterations) -- with ONE change: the training context is **512
tokens**, sampled as a random crop from each piece, instead of the full 2,048. Evaluation is on
whole test pieces, with NLL reported by POSITION BUCKET: 0-512 (in distribution), 512-1024 (1-2x
beyond) and 1024-2048 (2-4x beyond). Test pieces have median 912 and maximum 2,048 tokens, so the
last bucket is real but thin (28 of 77 pieces reach 1024, 7 reach 1536).

Same 2x2, 5 seeds: {index, path integration} x {RoPE, PoPE}.

## Registered verdicts

- **R1** In the 0-512 bucket, PoPE - RoPE should reproduce the in-distribution result already
  measured (-0.0322 NLL at full context). A different sign here means the shorter context changed
  the task, and nothing else in the run is interpretable.
- **R2 (the point of the run)** Does path integration's advantage GROW with position bucket?
  Registered prediction: **yes** -- MapWM - RoPE and MapPoPE - PoPE become more negative (better)
  from 0-512 to 1024-2048. This is the pattern in every other length-extrapolation result in this
  project. Tested as the paired per-bucket contrast with its MDE, and as the bucket-to-bucket
  change with its own MDE.
- **R3** PoPE's encoding advantage should hold at every bucket (it is a decoupling claim, not a
  length claim).
- **R4** If path integration's advantage does NOT grow, that is the third null for path integration
  on PoPE-paper data and is reported as such -- the registered conclusion becomes "path integration
  pays only where the task has explicit push/pop or move-by-k structure", with Dyck-2 the sole
  positive.

At n=5 the MDE is 1.25 sd. The in-distribution effect size for reference was 0.032 NLL with seed sd
0.010; extrapolation differences in this project are usually larger than in-distribution ones, so
this is better powered than the JSB run was, but a small effect will still read as unmeasured.

## Amendment 1 (2026-09-17) -- the rank test for MapPoPE's collapse

MapPoPE is the best arm in distribution (0.5235) and blows up past the training context (1.954 at
1-2x, 4.616 at 2-4x) while MapWM is the best arm out there (1.397). The full-context control shows
no collapse, so it is extrapolation-specific. The proposed mechanism: PoPE carries one phase per
ELEMENT (32 frequencies per head at d_model 256 / 8 heads) where RoPE-style rotation carries one per
PAIR (16), and path integration makes that phase an unbounded content-driven cumulative sum -- more
frequencies on an angle that grows without bound leaves the trained phase range sooner.

**Registered test**: the rank of the content-to-increment map bounds how much of the embedding can
drive the angle, so if this account is right the collapse should be DOSE-DEPENDENT in rank.
Arms added, 5 seeds each, everything else identical: **MapPoPE at r=1 and r=4**, plus **MapWM at r=1
and r=4 as the control row** (r=2 already run for both).

- **A1** MapPoPE far-bucket (1024-2048) NLL ordered r1 < r2 < r4. Confirmed if r1 - r2 is negative
  and detectable and r4 - r2 is positive and detectable.
- **A2** The same ordering must NOT appear (or must be much weaker) on the MapWM control row; if
  rank moves both rows equally the effect is about rank in general, not about PoPE's frequencies.
- **A3** If r=1 does not reduce the collapse, the rank account is REFUTED and the remaining suspect
  is the frequency count itself (32 vs 16), which rank cannot reach -- that would need a PoPE
  variant with pair-wise frequencies, which is not built and is not run here.

Note what this test cannot separate: rank bounds the SUBSPACE the increment lives in, not its
MAGNITUDE. A null on A1 leaves both "the frequency count is what matters" and "the angle magnitude
is what matters" alive; the latter would be tested by the omega base, which is untouched here.

## Amendment 2 (2026-09-17) -- the omega-base test for the MapPoPE collapse

Rank is refuted (amendment 1): the collapse survives r=1, r=2 and r=4 unchanged, and the rank effect
is equally present on the MapWM control row. The surviving suspect reachable by a knob is the ANGLE
MAGNITUDE. `PathIntegrator` sets omega_i = omega_max * (1/base)^(i/(n_b-1)) with omega_max = 2*pi, so
the base fixes how slow the slowest frequency is: a larger base spreads the schedule further and makes
the low-frequency blocks accumulate phase more slowly, which is what keeps an unbounded cumulative
angle inside its trained range for longer.

All runs so far used base = 2048 (the sequence length). Registered: **base in {512, 8192, 32768}**,
5 seeds, for **MapPoPE and MapWM both** (base 2048 already run for both), training context 512,
everything else identical.

- **B1** If angle magnitude drives the collapse, MapPoPE's far-bucket (1024-2048) NLL falls
  MONOTONICALLY as the base grows, and 32768 is detectably better than 2048.
- **B2** The control row must show a weaker effect. If MapWM moves by a similar amount, the base is
  again a general extrapolation knob and explains nothing specific about MapPoPE -- the same
  disconfirming logic that killed the rank account.
- **B3** If MapPoPE stays at 4.1-4.6 at every base, angle magnitude is refuted too, and the only
  remaining suspect is the FREQUENCY COUNT itself (one phase per element rather than per pair), which
  no hyperparameter here can reach -- it needs a pair-frequency PoPE variant, and the honest outcome
  is to say the mechanism is unidentified rather than build a variant to keep the story alive.
- Also reported: whether any base makes MapPoPE beat MapWM out of distribution at all.
