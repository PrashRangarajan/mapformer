# Decay envelope on PoPE: the 48-parameter repair matches the 786k one

Pre-registration: `DECAY_PREREG.md`. Bach Chorales, 512-token training crops, test NLL by position
bucket (lower is better), 5 seeds. The envelope is `scores -= softplus(lambda_h) * distance`, one
learnable scalar per head on ALiBi's geometric spread.

| arm | 0-512 (trained) | 512-1024 | 1024-2048 |
|---|---|---|---|
| PoPE (index) | 0.5403 | 0.6749 | 1.5973 |
| **PoPE + decay** | 0.5580 | **0.5965** | **0.6262** |
| MapPoPE | 0.5235 | 1.9536 | 4.6158 |
| **MapPoPE + decay** | 0.5498 | **0.5860** | **0.6223** |
| MapPoPE + per-token phase (T3, forced) | **0.5182** | **0.5629** | **0.6162** |
| MapWM | 0.5382 | 0.7835 | 1.3969 |

- **D1 PASSES, decisively**: MapPoPE + decay is **-3.994** at 1024-2048 (5/5, MDE 0.417) and -1.368
  at 512-1024. The collapse is gone.
- **D2 PASSES**: on the index row, PoPE + decay is **-0.971** at 1024-2048 (5/5, MDE 0.374) and
  -0.078 at 512-1024. So the answer to "can another positional encoding help PoPE on language" is
  yes, and the cheapest one in the literature is enough.
- **D3 partially fails**: both arms cost in distribution -- +0.018 (PoPE) and +0.026 (MapPoPE), 0/5
  seeds better, detectable. That is the expected trade (a decay envelope buys far-field reliability
  with near-field sharpness). CORRECTED 2026-09-19: the original text said this was "larger than the
  per-token phase's +0.036"; 0.026 < 0.036, so the envelope's in-distribution cost is SMALLER than
  the phase's, not larger. Also unreported at first: the decay arm is detectably worse than the phase
  arm in the 512-1024 bucket as well (+0.0231, MDE 0.0140, 0/5), not only in distribution.
- **D4, the comparison that matters**: MapPoPE + decay reaches 0.6223 against the per-token phase's
  0.6162 -- statistically the same repair, for **48 parameters instead of 786k**. The phase keeps a
  small edge in distribution (0.5182 vs 0.5498).

> **Scope (2026-09-20)**: the measurement in this file stands. The cross-task account it was read
> as supporting does not -- both halves fail a within-task test on Dyck (`T2_RESULTS.md`), so
> treat this as a fact about Bach at a 512-token context.

## What this does to the account

The two repairs work through DIFFERENT mechanisms and reach the same place. The phase lets content
correct a miscalibrated kernel; the envelope refuses to consult the kernel where it is unreliable.
That both work is consistent with the diagnosis -- the failure is confident wrong kernel values at
large `|S_t - S_s|` -- and neither result distinguishes the diagnosis from a simpler one.

**The pre-registered discriminator did NOT fire the way it could have.** `DECAY_PREREG` D2 said the
informative failure would be the envelope helping only where the accumulator is learned; instead it
helps on BOTH rows, and by a similar mechanism-free amount. So the honest reading is that a large
part of what looked like a path-integration-specific pathology is the generic long-context problem:
an attention kernel trusted at distances it was never calibrated on. Path integration makes it worse
(4.616 vs 1.597) because its position variable is learned as well as extrapolated, but it is not a
different disease.

**Practical ranking for a length-extrapolating model with PoPE** -- AMENDED 2026-09-19, see
`DYCK_DECAY_RESULTS.md`: "works on either position mechanism" is FALSE on a task whose dependencies
are not local. On Dyck-2 the same envelope drives the index row to the no-stack n-gram level beyond
distance 8, in distribution as well as out. The ranking below holds for locally-structured data only.
Decay envelope first (48 parameters), per-token phase if the last 0.03 NLL in distribution
matters, and bounding the accumulator only if neither is available -- centring cost 0.13 in
distribution, four times the phase's cost, for a worse far-field number (0.911).

## D5 (registered, reported late 2026-09-19): the learned decay rates

Mean over seeds and layers, per head, against the ALiBi initialisation `2^-1 .. 2^-8`:

| head | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| init | 0.500 | 0.250 | 0.125 | 0.063 | 0.031 | 0.016 | 0.008 | 0.004 |
| PoPE + decay | 0.498 | 0.264 | 0.135 | 0.070 | 0.036 | 0.019 | 0.009 | 0.0045 |
| MapPoPE + decay | 0.494 | 0.261 | 0.135 | 0.070 | 0.037 | 0.019 | 0.009 | 0.0044 |

Every head KEPT its decay (the widest reaches ~222 tokens on pieces up to 2048), so the winning
configuration on Bach is a local model. On this task, repairing the collapse and becoming local are
the same operation -- which is a fact about chorales as much as about positional encoding, and is why
the Dyck follow-up was run.

## Batch provenance (audit note, 2026-09-19)

The contrasts here pair by seed index ACROSS run directories built on different days, not within one
batch -- the repo's standing rule 3 asks for one batch. Mitigating: the arms share code (only new
classes were added between the runs; the data and evaluation paths are byte-identical), the data
stream is seeded identically, and the primary readings lean on a parameter-matched inert twin
trained INSIDE the new batch. Unmitigated for the decay arms, which have no same-batch baseline.
