# Autonomous session, 2026-10-12 (user away; GPU 0 only, another user held GPU 1)

All batches pre-registered, independently checked before launch (rule 29), amended where the check found problems, and
written up with registered verdicts first. Nothing is running. 24 commits since the last push (a16d178), not pushed.

## The question
Paper framing (user, 2026-10-11): MapFormer's phase as Howard et al. 2014's integrator -- a constant rate (time, index RoPE),
a non-negative rate (a content clock) or a signed rate (position, a map). Prior art (`docs/lit/LIT_CLOCK_MAP.md`,
`LIT_HOWARD_TRANSFORMER.md`): the dichotomy is Howard's; the mapping of his three cases onto the three rotary-position families
was not found stated anywhere (one targeted search).

## The main finding: in the rotary slot, sign is learnability, not representation
The RING_DIAL audit showed (by construction) that a non-negative ROTARY step can represent any signed code: angles wrap mod
2 pi, so a second token's step 2 pi / omega - Delta reproduces -Delta. So Howard's Cases II and III coincide in what a rotary
phase can represent. What separates them is what training finds:
| result | file | status |
|---|---|---|
| On the 1D ring, the non-negative step converges on 2/8 and 1/8 seeds vs 8/8 for the signed step where steps cancel (p_plus 0.5 / 0.75); unmeasured at 0.9 | `RING_DIAL_RESULTS.md` | registered |
| Three times the training does not help: 3/8 at 900 vs 3/8 at 300 epochs (same seeds) | `RING_BUDGET_RESULTS.md` | registered (PARTIAL) |
| No non-negative run, solved or not, finds the wrap solution: 2-11% of channels cancel mod 2 pi vs 82% (ring) and 98% (torus) for signed | same + `docs/audits/2026-10-12/torus_wrap_probe_out.txt` | secondary / post hoc |
| Same measure on both tasks (loss AUC, paired by seed): signed learns the ring map faster on 8/8 seeds at every p (+0.23 to +0.35, p 0.0078) | `docs/audits/2026-10-12/ring_auc_companion_out.txt` | post hoc |
| On a content-counting task (k-back recency) the non-negative step learns faster on 12/16 seeds (AUC -0.197) but p 0.053: UNMEASURED | `CLOCK_SPEED_RESULTS.md` | registered |
| An earlier accuracy test of the same question was unmeasured (nothing converged in 300 epochs); index RoPE fails the task | `CLOCK_SIDE_RESULTS.md` | registered |
Read: an **asymmetric learnability effect** -- a signed rotary step is clearly easier to train into a map; a non-negative
one trends easier to train into a content clock, not established. No double dissociation is claimed.

## Corrections made today
- "A monotone rotary step cannot represent a -1 action" (CLAUDE.md sign row, memory, claim map, lit note, SIGN_MATCHED
  reading): false; the sign result is learnability.
- The often-quoted clock/map crossover (-0.280 torus, -0.004 recency) was out of distribution on both sides (CLAUDE.md row
  corrected). TinyStories and code models are pure clocks (growth exponent ~1.0, `docs/audits/2026-10-12/alpha_language_out.txt`).
- Process slips, each disclosed in its results file: a CLOCK_SIDE secondary crashed after the registered lines printed (fixed,
  marker set by hand); the CLOCK_SIDE calibration chose a difficulty by accuracy level, not convergence, so nothing converged.

## Where the paper stands
`docs/paper/CLAIM_MAP_clock_map.md` maps every claim to its file and status. Candidate core: (1) Howard's three cases
<-> the three rotary families, with the twist that in the rotary slot Cases II and III coincide representationally;
(2) the learnability asymmetry above; (3) depth substitution and navigation (in words too) at matched length; (4) natural text
learns a clock (TinyStories, code), with no gain over the index at our scale (CoPE shows gains at larger scale: scope it).

## Open, for the user
- Push the 24 commits.
- Whether to firm up the clock side (CLOCK_SPEED at more seeds or a converging budget: ~3-7 h on one GPU) or write.
- The baseline table (~8-10 h on both GPUs) is still the first thing a reviewer will ask for.
