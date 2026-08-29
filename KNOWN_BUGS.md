# Known bugs and traps (found by review 2026-08-27/28, most NOT yet fixed)

Ordered by how much damage they can do silently. Everything here was found by
reading code, not by anything failing loudly — which is the point.

## SILENT — will corrupt a comparison without any error

**1. `MapPoPE-Hier` / `PoPE-Hier` / `Plain-Hier` silently ignore `bottleneck_r`.**
Verified: r=2 and r=4 give IDENTICAL param counts (9,724,672), so `**kwargs` swallows
it. `MapWM-Hier` and `MapWM-FlatHG` DO accept it (28,368,376 -> 28,371,016). Any
hier-vs-flat comparison in the PoPE family therefore confounds RANK, unlogged.
*Fix:* give every wrapper an explicit `bottleneck_r=2`, or try/except the kwarg
instead of `inspect.signature` (which cannot see through `*args/**kwargs`).

**2. `train_hourglass_enwik8.build()` hardcodes dim/heads/n_layers for the three
plain baselines.** `HourglassPlainLM(dim=512, heads=8)` / `FlatPlainLM(n_layers=10)`
ignore the arguments passed in. Put `flat9` in a sweep with `--dim 256` and it
silently trains at 512 against 256-dim MapFormer arms.
*Fix:* pass the arguments through.

**3. Misaligned seed pairing in `run_mw_hier_ablation.sh`'s aggregator.** `h` and `f`
are each filtered for `None` independently, then zipped. If Hier is missing seed 1 and
FlatHG is missing seed 2, both lists are length 2, the length check passes, and it
pairs Hier-s0 with FlatHG-s0 and **Hier-s2 with FlatHG-s1**.
*Fix:* build pairs keyed by seed before zipping.

**4. `agg_miniworld.py`'s bare `except` can disable the convergence gate entirely.**
Any `torch.load` failure returns `None`, the arm is never added to `stuck`, and
`complete` passes with zero conditioning while the header still advertises it.
*Fix:* track load failures in a third list and include it in the predicate.

**5. Probes recompute `cumsum(delta) * omega` inline** instead of calling
`path_integrator(delta)` — `probe_position_decode.py:61`, `probe_goal_distance_state.py:99`,
`model_inekf_*`. Run against a **ConvDelta** checkpoint they silently report the
UNFILTERED path integral. (`ap_kernel_diagnostic.py` calls the module and is fine.)

## FIXED this session

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
