# MiniWorld fixed-map factorial — findings (2026-08-25)

Full multi-seed factorial: {Vanilla, MapPoPE-Flat}=path-int × {RoPE, PoPE-Flat}=index
× {raw, allocentric} × 3 seeds = 24 arms, all converged (train loss < 0.16, no
non-convergence flags), d=256 / 4-layer / 100 epochs. Task = path integration on a
KNOWN map (fixed obs_map per seed, novel walk per episode), scored on cross-cell
revisits, non-blank accuracy (chance 0.0625, oracle 1.0). Tables auto-generated in
`MINIWORLD_FIXED_RESULTS.md` (T=512) and `MINIWORLD_FIXED_RESULTS_T1024.md`.

## 1. MiniWorld is now LEARNABLE (the primary goal)

Held-out non-blank accuracy, per arm (T=512): 0.62–0.82, vs the fresh-map version's
0.18–0.27 (which memorised its 3k-traj buffer: train loss 0.2 but held-out NLL 6.2).
The fix was a TASK redesign, not just a bigger model: fresh-map-per-episode demands
in-context map BUILDING (needs ~infinite maps); fixed-map isolates path integration
on a known map, which the same buffer supports. All 3 seeds learn, not just seed 0.

## 2. The MiniGrid allocentric FLIP does NOT reproduce

Position effect = (path-int mean) − (index mean), paired within seed:

| length | raw | allocentric | per-seed Δ(allo−raw) |
|---|---|---|---|
| T=512  | +0.020 ± 0.013 | +0.008 ± 0.009 | +0.002, −0.035, −0.003 |
| T=1024 | +0.010 ± 0.017 | −0.021 ± 0.013 | −0.049, +0.002, −0.045 |

On MiniGrid the effect flipped −0.02 (raw) → +0.02 (allocentric). Here it does the
OPPOSITE if anything: allocentric slightly LOWERS the path-int−index gap, and at OOD
length drives it negative. No flip at either length.

Contrast with MiniGrid's raw regime: there raw was NEGATIVE (the cumsum genuinely
cannot integrate turn-then-forward, so path-int LOST to index). Here raw is small-
POSITIVE (+0.020): even with un-integrable raw actions, the path-int arm is not
disadvantaged. That difference is the crux (see §4).

## 3. Allocentric helps ALL architectures ~equally (+0.15), via input not mechanism

Vanilla 0.653→0.801, MapPoPE 0.662→0.819, RoPE 0.620→0.798, PoPE 0.655→0.807
(raw→allo, T=512). Allocentric displacement-direction tokens (25 classes) are simply
more informative INPUT than raw turn/forward (3 classes) — every arm attends to them,
so index arms benefit as much as path-int arms. This is NOT the cumsum path-
integration mechanism being restored; it is a richer observation channel. The n-gram
gate confirms this is not a trivial-localisation leak (allo non-blank n-gram 0.140 <
marginal 0.150, accuracy DECREASES with context order — `MINIWORLD_GATES_ALLO.md`).

## 4. Why the position effect is small in BOTH encodings (interpretation)

The index arms (RoPE/PoPE) still receive the ACTION TOKENS as input; a transformer
integrates position from them via ATTENTION, without the explicit ω·cumsum(Δ)
rotation. So on a learnable fixed map at T=512–1024, attention already localises well,
and the explicit path-integration code is only a weak inductive bias worth ~+0.02 —
not the decisive factor. This matches the project's established finding that "a plain
transformer path-integrates via attention; MapFormer's SO(2) code is an inductive
bias, not privileged info" (compositional / family-tree sessions), and the torus's
large +0.461 position effect must come from a regime where attention CANNOT integrate
(longer/harder than tested here). The MiniGrid flip likewise needs a regime where the
cumsum is load-bearing; fixed-map MiniWorld at these lengths is not that regime.

## 5. Honest caveats

- Effect sizes are small (±0.02), near the n=3 noise floor. The right read is "no
  reliable flip; the position effect is near zero and encoding-insensitive", not a
  strong negative about path integration in general.
- Fixed-map deliberately dilutes the cumsum's role (a known map + attention suffice).
  A regime where index truly fails (much longer OOD, or fresh-map done right at scale)
  might still show a flip — untested, and expensive (fresh-map needs ~infinite maps).

## 6. Implication for the Habitat plan

The hoped-for mechanism — allocentric recoding SPECIFICALLY rescuing path integration
— does not appear on continuous-3D fixed-map navigation. Before committing to Habitat,
the open question is whether ANY tested regime here makes the explicit path-
integration code decisive (large position effect). On MiniWorld fixed-map it is not.
Allocentric recoding is still a real, free win (+0.15 to every architecture), but as
better input, not as a path-integration fix. Porting to Habitat should be justified by
a regime where the position effect is large, not by the flip hypothesis, which this
factorial does not support.
