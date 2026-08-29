# Known bugs and traps (found by review 2026-08-27/28, most NOT yet fixed)

Ordered by how much damage they can do silently. Everything here was found by
reading code, not by anything failing loudly — which is the point.

## SILENT — will corrupt a comparison without any error

**(5 of the original 5 are now FIXED — see below. Nothing is open in this section.)**

## FIXED (2026-08-27/28)

- **Non-deterministic val in `train_hourglass_enwik8`** — val batches were drawn from
  the global RNG, resampled per checkpoint AND differing per model. Swings 0.02-0.07
  against an 0.011 spread; min-over-last-6 REORDERED the arms. Now a fixed generator;
  verified bit-identical across two independent runs. **This invalidated the entire
  12k language result.**
- **Stale-JSON skip guard** — the trainer wrote the FINAL filename at every eval, so a
  1k-iter partial looked like a finished 36k run to a `-f` test. Periodic dump now goes
  to `.partial.json`; the guard checks for `wall_total_s`.
- **`validate_miniworld` G7 oracle read a stale `obs_map`** on fresh-map (redrawn each
  episode) — spurious FAIL at 0.27 when the true oracle is 1.0.
- **`flops_proxy` wrote a bare `NaN`** into JSON, which strict parsers reject. Now `None`.
- **`experiment_audit.py` control matching** used equality where variant keys carry an
  `_oracle` suffix — silently reported "no pairs found" and never measured the floor.

### Fixed 2026-08-28 (the five silent ones), each with a verification

- **PoPE hourglass family swallowed `bottleneck_r`.** Root cause was NOT the call
  site's `inspect.signature` as first diagnosed — those classes accept the kwarg
  without error, then `_widen_to_d(self, kw.get("grid_size", 64))` REBUILDS
  `action_to_lie` at the default rank, discarding it. Now named parameters on
  `MapFormerWM_Hourglass_PoPE` / `_CoarseIdx`. *Verified:* r=2 vs r=4 param counts
  now differ (2,503,168 / 2,504,192) where they were identical.
  Same line also silently reset `grid_size` to 64 if passed positionally — also fixed;
  *verified* omega_min = 2π/grid_size at grid 64 and 512.
  `PoPE-Hier` remains correctly r-invariant (it has no action subspace); documented
  in-class so nobody "fixes" it.
- **`build()` hardcoded dim/heads/n_layers for the plain baselines.** Passed through.
  *Verified:* flat9 is 7.2M at dim 256 and 28.6M at dim 512 where it was 28.6M for both.
- **`build()` now FAILS LOUDLY** if a non-default rank was requested and the class
  cannot honour it, instead of training the wrong model quietly.
- **Misaligned seed pairing in `run_mw_hier_ablation.sh`.** Pairs are built keyed by
  seed before filtering, so a seed counts only when both arms have it.
- **`agg_miniworld.py`'s bare `except` disabled the convergence gate.** An unreadable
  checkpoint now returns its own signal, is listed separately, and counts against
  `complete`. *Verified* with a deliberately corrupted `.pt`: reports
  `UNREADABLE ... (UnpicklingError)` and withholds the verdict, where it previously
  passed the gate silently.

**Still open: #5, probes recompute `cumsum(delta) * omega` inline**
(`probe_position_decode.py:61`, `probe_goal_distance_state.py:99`, `model_inekf_*`)
instead of calling `path_integrator(delta)`. Against a **ConvDelta** checkpoint they
silently report the UNFILTERED path integral. (`ap_kernel_diagnostic.py` is fine.)

## TRAPS (not bugs, but they have cost time)

- **Hourglass variants IGNORE `--n-layers`** and always use their own 3-block scaffold.
  At `--n-layers 4, dim 512` that is 2.38M vs a flat model's 3.17M — a 33% capacity
  confound with nothing in the log to reveal it. Size by `dim` instead: at dim 880 all
  hier variants land within 0.02% of each other.
- **`_geometry_bounds` takes a bounding box over rooms.** Exact for OneRoom; WRONG for
  any non-rectangular multi-room env (DoorKey, MultiRoom), where it would include
  unreachable cells. Fix with point-in-room assignment before leaving OneRoom.
- **Oracle clamp-rate counters never fire in the real pipeline** — they are per-instance
  and trajectories are built in spawned workers, so the parent's counters stay 0 and the
  guard skips the print. Measure it in a standalone single-process probe.
- **All aggregators append with `>>`.** A re-run stacks a second, potentially
  contradicting report into the same `.md`.
- **`enwik8` output filenames encode only `{model}{tag}`** — not seed, dim, lr or rank.
  Two runs differing only in `--seed` into the same `--out` overwrite each other, and
  the survivor looks perfectly valid.
- **Summary `.md` files drift from the raw data.** Three cases found in two days: a
  threshold mixed 0.2/0.4 within one paragraph; a stale 7-arm JSON whose `.md` rendered
  8 rows; and an enwik8 hierarchy figure (2.00 vs 2.07, hierarchy better) that NO saved
  data supports — the real numbers are 1.4844 vs 1.4727, hierarchy WORSE. Generate
  headline numbers from the JSONs rather than typing them.
- **Aggregator `--out` defaults to a TRACKED results file.** `agg_miniworld.py`
  defaults to `MINIWORLD_FIXED_RESULTS.md`, so pointing it at any other `--runs-dir`
  — including a throwaway smoke test — overwrites the canonical result in place with
  no warning. Hit while testing the fix above; caught only by `git status`. Pass an
  explicit `--out` for anything that is not the real run, and check `git status`
  after running an aggregator.
