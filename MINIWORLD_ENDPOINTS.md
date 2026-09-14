# MiniWorld map-extent endpoints: the per-seed contrasts behind +0.305 and +0.015

`VISITS_TEST.md` (post-hoc pooled table) and `CLAUDE.md` quote two endpoints that had no results
file, sd or MDE. This file computes them from the stored per-seed JSON and checkpoints, with the
same pairing and readout as `agg_alias.py`: path-integrated (`Vanilla`) minus index (`RoPE`),
paired by seed, non-blank accuracy at T=512, held-out ("oracle") evaluation. Nothing was retrained.
Generated 2026-09-13.

| condition | source dir | epochs | contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|---|---|
| grid 32, n_obs 256 (2 cells/token, 512 occupied) | `runs/alias_follow/n256_800` | 800 | path-int - index | +0.305 | 0.048 | 0.077 | 3/3 | DETECTABLE |
| grid 16, n_obs 64 (2 cells/token, 128 occupied) | `runs/alias_follow/g16` | 400 | path-int - index | +0.015 | 0.013 | 0.020 | 3/3 | unmeasured |

Per seed (nb_acc T=512 / final loss / loss slope over the last 10% of epochs):

| condition | seed | Vanilla | RoPE |
|---|---|---|---|
| grid 32, n_obs 256 | 0 | 0.998 / 0.0156 / -1.8e-04 | 0.655 / 0.3981 / -2.5e-04 |
| | 1 | 0.947 / 0.0102 / -6.0e-05 | 0.696 / 0.3440 / -1.7e-04 |
| | 2 | 0.997 / 0.0157 / -4.1e-05 | 0.677 / 0.3712 / -2.7e-04 |
| grid 16, n_obs 64 | 0 | 1.000 / 0.0022 / -6.0e-05 | 0.992 / 0.1317 / -3.3e-04 |
| | 1 | 1.000 / 0.0016 / -1.7e-05 | 0.971 / 0.1439 / -3.2e-04 |
| | 2 | 1.000 / 0.0188 / -3.1e-04 | 0.994 / 0.1498 / -4.7e-04 |

Every arm is under `agg_alias.py`'s flatness threshold (|slope| < 5e-4); RoPE grid 16 seed 2 is
the closest (-4.7e-04).

## Caveats that travel with these numbers

- **n=3 per condition.** By this project's convention these are exploratory, whatever the
  verdict column says.
- **Grid 16 is a ceiling, not an informative null.** The index arm scores 0.971-0.994, so there is
  almost no room for path integration to add anything. It is the same ceiling reading
  `VISITS_TEST.md` gives condition B.
- **The two endpoints were trained at different budgets** (800 vs 400 epochs), so the threshold
  comparison between them mixes map size with budget.
- **MDE and the noise floor are different tests.** "Detectable" here compares against the paired
  MDE. The project's older measured noise floor (0.150, from unconverged function-identical twins)
  is a separate and more conservative bar; +0.305 clears both, and +0.015 clears neither.
