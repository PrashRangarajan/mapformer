# Pre-registration: is Dyck's position effect DEPTH-SUBSTITUTION, and is it CONFOUNDED with the frequency ladder?

Written before any run in this batch. Both questions came out of an agent audit
that also retracted the code OOD result (`CODE_RESULTS.md`).

## Why

The surviving positive claim in the language line is Dyck's position effect. Two
things make it weaker than published here:

1. **It was quoted on F1, which this project's own rules forbid as a headline**
   (no-stack floor 0.884-0.904). Under Hewitt distance-averaged closing accuracy
   the position main effect HALVES, +0.306 -> **+0.136**, and the interaction
   goes from +0.009 to **+0.065**, the size of the encoding effect.
2. **Dyck has no 2-layer path-integrated arm anywhere.** All MapWM/MapEM/MapPoPE
   runs are 1-layer. Meanwhile the index arms at 2 layers already close most of
   the gap: at the training cell RoPE-2L 0.914 / PoPE-2L 0.919 against MapWM-1L
   0.992, and at the hardest OOD distance bucket (d>=33) **RoPE-2L 0.592 reaches
   MapWM-1L 0.611 and PoPE-2L 0.638 beats it**. This repo's own
   `HORIZON_RESULTS.md` says attention horizon is CAPACITY, not architecture
   (~2 steps at 1 layer, ~32 at 4 layers), which puts Dyck's stack distances
   squarely in the range depth buys.

So "path integration is decisive on Dyck" may be a statement about **1-layer
models**, not about Dyck.

## E1 -- the depth-matched 2x2 (never run)

`MapWM-2L_r2`, `MapPoPE-2L_r2`, `RoPE-2L`, `PoPE-2L`, 2 layers / 2 heads / d128,
**all four retrained in ONE batch** (rule 3: never compare fresh arms to stored
checkpoints), 8 seeds. Parameter parity verified before launch: 398,533 /
398,981 / 398,085 / 398,533-ish -- within 0.22%.

**Primary readout, fixed now: Hewitt distance-averaged closing accuracy at
L128 D12** (chance 0.500), with closer accuracy at **distance >= 33** as the
co-primary, and F1 plus invalid-probability mass reported alongside and never
selectively. MDE = 2.8*sd/sqrt(8) per paired contrast.

- **F1 (kills the claim's scope).** If `position(2L)` is below its MDE while
  `position(1L)` = +0.136 is above it, **Dyck's position effect is
  depth-substitution**, and the divergence from code needs no further
  explanation: the two batches measured a capacity bottleneck and a kernel
  bottleneck respectively.
- **F2.** If `position(2L) ~ position(1L)`, depth is dead and the effect is
  genuinely the position mechanism.
- **Registered prediction** (from the stored per-distance table, so this is a
  real prediction and not a hedge): position shrinks to within MDE on the RoPE
  row and survives on the PoPE row -- i.e. a **detectable interaction**, which
  the 1-layer F1 2x2 says is absent. Falsifiable twice.

## E2 -- the frequency-ladder control (never run)

The index classes take their rotary ladder from `base` (default **10000.0**) and
ignore `grid_size`, while the path arms derive theirs from `grid_size` (=32 on
Dyck). Measured from the trained checkpoints: path arms' slowest wavelength
**143-284 tokens**, index arms' **47,116** -- a 200-300x mismatch riding on the
"position" axis, with ~34% of index channels never completing a cycle inside the
eval window. `ROPE_CANONICAL.md` tested the schedule's SHAPE at fixed base 10000,
never the base.

`RoPE-1L` and `PoPE-1L` at `--rope-base` in {32, 128, 10000}, 8 seeds each. The
10000 arm is a FRESH control in the same batch, not the stored one.
`--rope-base None` was verified byte-inert (identical `inv_freq`) before launch,
so no stored arm changes.

- **F3.** If index@base32 or @base128 closes **>= half** the position gap at
  L128 D12 on the primary metric, then a large part of "path integration on
  Dyck" is spectral allocation rather than content-dependence -- and the same
  confound sits in the code batch at a 20x ratio.
- **F4.** If it closes none, the confound is dead and every position claim in
  this line gets stronger. My guess: base 32 helps in distribution and aliases
  at L128, i.e. the effect is real but smaller than published.

## Cost and voiding

80 runs at ~55 s each (Dyck arms are ~51k-399k params). Void if any arm is
unconverged (registered slope check, -0.005/1k), if arms are split across
batches, or if a gate regresses.
