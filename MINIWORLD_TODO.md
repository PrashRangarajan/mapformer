# MiniWorld — deferred fixes (apply AFTER the running factorial finishes)

Code review (2026-08-25) findings. **Do NOT edit `miniworld_env.py` or
`train_miniworld.py` while `run_miniworld.sh` is still spawning arms** — later
arms would import changed code and the within-batch comparison becomes a
between-code one (CLAUDE.md 2026-08-19 rule). These land in the Habitat-track
config, which will retrain ALL arms in one batch so no consistency issue.

## Resolved already (safe edits, done)
- [x] #1 allo n-gram gate: added `--allocentric` to `validate_miniworld.py`;
      ran it (fixed-map, G=8). ALLO stream PASSES — non-blank n-gram 0.140 <
      marginal 0.150, accuracy DECREASES with order → no localization leak.
      Result in `MINIWORLD_GATES_ALLO.md`. Standing Rule 1 satisfied both streams.
- [x] #2 aggregator hardened: `agg_miniworld.py` now reads final train loss from
      each `.pt`, flags arms with loss > 0.6 (⚠), reports the position effect
      PER SEED paired within seed (mean±std), and WITHHOLDS the flip verdict on
      any missing/non-converged arm.

## Deferred (edit modules — do after factorial done)
- [ ] #3 EVAL BUFFER CACHE (biggest throughput win). `train_miniworld.py::evaluate`
      regenerates 128 live MiniWorld trajectories per length (T=512 AND T=1024),
      and they are IDENTICAL across all 4 variants at a given (seed, enc) because
      the walk is model-independent and seeded `RandomState(5000+seed)`. Build a
      disk-cached eval buffer keyed by (env, seed, enc, T), shared by all 4
      variants. Removes ~7/8 of eval cost AND makes the effect exactly paired.
- [ ] #4 BOUNDS FROM GEOMETRY. `_measure_bounds` runs a 4000-step rendered random
      walk on every env construction (both env and env_test, all 24 procs, even
      buffer-load-only runs). Read OneRoom room min/max x,z directly instead —
      exact, zero cost, and removes the clamp bias (directed walk reaches beyond
      the random-walk-sampled bounds → outer ring collapses onto border cells).
      Failing that: percentile + generous margin, cached by (env_name, seed).
- [ ] #5 SHRINK RENDER + EARLY MACRO BREAK. RGB obs is discarded (obs comes from
      obs_map) but every `u.step` still renders POV at 80x60. Pass small
      `obs_width`/`obs_height` to `gym.make`. Also break the forward macro early
      when position stops changing (wall-blocked) instead of spinning to cap≈19.
- [ ] #6 CACHE KEY. `build_or_load_buffer` key omits `p_empty` (collision risk)
      and any code/policy version — a directed-walk / `_disp_dir` / `_macro` edit
      will silently reuse stale buffers. Add `p_empty` + a `CODE_VERSION` token
      bumped on any data-affecting edit.

## Minor / hygiene
- [ ] #8 dead `_sample_action` (miniworld_env.py); `sched.step()` skipped on the
      `if m.sum()==0: continue` branch (LR under-steps, negligible at T=512);
      `agg_miniworld` default --runs-dir still points at the stale `runs/miniworld`
      (launcher overrides, but a bare invocation aggregates stale fresh-map data).
- G7 oracle in the validator is tautological (recomputes obs with the same
  `_cell`+`obs_map`) — a consistency check, not an independent answerable-from-
  position test. Fine as-is, just don't over-read it.

## Non-issues confirmed by review (do not "fix")
- raw vs allo buffers share IDENTICAL walks+obs_map (allocentric only selects
  which token is appended; consumes zero RNG) → input-matched, RNG bit-identical.
- vocab differs (raw 20 vs allo 42) but effect is path-int−index WITHIN an
  encoding → not a confound. Token-id layout collision-free.
