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

## Literature validation (2026-08-25) — how MiniWorld is actually run
Sources: MiniWorld troubleshooting doc + source, Issues #16/#44, JAXenstein
(arXiv:2605.19926), arXiv:2401.05946, Minari.
- Headless native pyglet-EGL, render_mode=None: STANDARD/optimal (better than
  xvfb on a GPU box; NVIDIA/xvfb conflict is documented). Ours is the pyglet-2.x
  native-EGL form (newer than the doc's PYOPENGL_PLATFORM=egl + pyglet==1.5.11).
- render_obs no-op: NECESSARY, not a hack. MiniWorld calls render_obs() every
  step()/reset() UNCONDITIONALLY; there is NO state-only mode. Cleaner FORM =
  subclass-override instead of class monkeypatch (functionally identical). Our
  disable removes exactly the bottleneck the field complains about (MiniWorld
  ~1.8-2k steps/s single-core is a documented UPPER bound; GL contexts don't
  fork -> the "24 workers serialize" symptom is a known limit, Issues #16/#44).
  The field's real fix is JAX rewrites (NAVIX, XLand-MiniGrid, JAXenstein); for
  us (pixels discarded) disabling render is the correct win.
- Disk-cached token buffer: equivalent to standard offline practice (Minari is
  the community format; we store tokens/positions not pixels -> already optimal).
- WRITEUP-HONESTY FLAGS (state explicitly; reviewers know MiniWorld):
  * Using MiniWorld as a GEOMETRY generator with a location-keyed aliased token
    instead of its RGB is UNUSUAL for MiniWorld specifically (MiniWorld has no
    symbolic-obs mode; MiniGrid's SymbolicObsWrapper does). The PARADIGM (discrete
    tokens for cognitive-map sequence models) is standard (TEM/CSCG), and
    arXiv:2401.05946 does nearly our design on DeepMind Lab (k-means k=128 tokens,
    2048 random walks len 400) -- but they quantize PIXELS; we key to location.
    Cite 2401.05946 as precedent; it also validates PERCEPTION_EXPERIMENT_PLAN
    Phase B (VQ/k-means frame tokens).
  * The directed-walk policy and the "macro forward until the cell changes" are
    OUR abstractions (random is MiniWorld's convention; action-repeat/biased-
    exploration-for-coverage are recognized techniques). Describe as such.

## NEXT-RUN efficiency (2026-08-25 efficiency review; apply AFTER current run)
Current: 3.18M params, BS=24, L=1023, 4 layers, ~14GB/24GB at 2 arms/GPU, both
GPUs 100% (compute-bound). Already efficient (DO NOT touch): parallel build,
render stub, geometry bounds, eval cache, cache key, 4 layers (HORIZON_RESULTS
justifies depth). Wins, by class:
- RESULT-INVARIANT (apply freely next run):
  [ ] #4 move the whole buffer to GPU ONCE (196MB int64 -> store int16 = 49MB,
      .long() on gather) -- kills per-step tok_t[idx].to(dev) H2D copy + CPU
      gather; exact same integers. train_miniworld.py:207.
  [ ] #5 accumulate loss.detach() on GPU, .item() ONCE per epoch (not per step)
      -- removes 18k CUDA syncs. train_miniworld.py:215.
- RESULT-CHANGING (bigger; rebaseline ALL arms together, NEVER mid-comparison):
  [ ] #1 F.scaled_dot_product_attention(is_causal=True) instead of hand-rolled
      matmul/softmax in model.py:226 -- ~1.3-1.8x AND big memory drop (the
      402MB x4 scores tensors are why 2 arms eat 14GB); enables 4+ arms/GPU.
      Flash backend changes FP accumulation -> tiny numeric drift.
  [ ] #2 torch.autocast('cuda', bfloat16) around forward+loss -- ~1.5-2x on 4090
      (no GradScaler for bf16). Numeric drift.
  [ ] #3 torch.compile(model) -- ~1.2-1.5x, static shapes, ~invariant.
  Combined #1+#2+#3 ~= 2-3x/arm. BS and layer count must NOT change within a
  comparison.

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

## FRESH-MAP FACTORIAL — design spec (from 2nd code review, 2026-08-25)

The fixed-map factorial found NO flip, but on a FIXED map path integration is not
load-bearing (memorised map + attention supply coarse position). The flip should
live in the FRESH-MAP (in-context) regime. Two review agents vetted the plan.

### Code review = clean (no hard bugs). Confirmed correct:
macro early-break is data-invariant (fwd macro never turns -> persistent freeze,
sliding gives nonzero motion); chunk-split sums exactly; worker seeds collision-
free (prime 100003 >> 24); spawn avoids EGL dup; v2 cache-key bump correct.

### Must-fix BEFORE / IN the fresh-map factorial (validity agent F1-F5, code F2-F3):
- [ ] **F1 (crux): include the hypothesised WINNER + a solvability ceiling.** A null
      is uninterpretable ("attention suffices" vs "under-data") unless >=1 arm
      DEMONSTRABLY solves the held-out task. Train allo arms too (predicted winner);
      report a positional-oracle upper bound. Rule 5.
- [ ] **F2: context-destruction ablation gates any claim.** On trained models, shuffle
      in-context obs tokens (and separately the action stream) -> nb_acc must collapse
      to ~0.0625. (Made Match-Query citable 0.918->0.074; voided hier-goal 0.912->0.913.)
- [ ] **F3: re-run the n-gram gate on the FRESH-map stream, BOTH encodings, BEFORE
      training.** Existing gates were fixed-map. `validate_miniworld.py` supports it
      (drop --fixed-map, add/omit --allocentric). Gate-before-training rule.
- [ ] **F4: lead go/no-go on held-out NLL < ~2.77** (below chance = genuine in-context
      learning), nb_acc secondary (0.27 bar is weak, ~4x chance). Also measure the
      memorisation ceiling at the SAME 24k/d256/nl4 config for an apples bar.
- [ ] **F5: probe epochs must match the factorial (100, not 40).** A 40ep null could be
      undertraining (bit this project 3x).
- [ ] **Code F3 (shared-key race): serial pre-build every unique (seed,allo) buffer
      FIRST**, then launch training arms (as run_mw_fresh_probe.sh does). run_mw_missing.sh
      lacked this; on a cold cache two shared-key arms race to write the same .pkl.
- [ ] **Per-GPU <=2 cap** (not total) — the OOM bug that killed 7 arms.

### Deferred (not needed for OneRoom):
- Code F1: `_geometry_bounds` bounding-box is wrong for NON-rectangular multi-room
  envs (DoorKey/MultiRoom) — includes unreachable cells. Fix with point-in-room
  assignment when we leave OneRoom. Also guard rooms with axis < 2*radius.
- Code F2 (cache key): add `w{n_workers}` (or par/ser tag) — content depends on
  worker count; repro hazard only if a buffer is deleted+rebuilt with a different
  --n-workers. Cheap; apply when editing train_miniworld next.

## Fixed 2026-08-25 (fresh-map run)
- BIG WIN: disabled the wasted POV render (miniworld_env.py __init__ stubs
  self.u.render_obs). MiniWorld renders the camera on every u.step via EGL/GPU,
  but we discard the image (obs = obs_map token). Render runs AFTER physics and
  touches neither agent.pos nor RNG -> DATA-INVARIANT (verified: positions
  identical with/without). Speedup: 476 -> 42 ms/traj single-process (11x); a 24k
  buffer 2727s -> 74s at 24 workers (37x). Root cause of the old slowness: 24
  workers all rendering on ONE GPU serialized the EGL streams. NOT a version bump
  (data identical) -> old render-on buffers reused as-is. If a future variant
  needs pixels (PERCEPTION_EXPERIMENT_PLAN.md), gate the stub behind a flag.
- validate_miniworld.py G7 oracle: was reading `w.obs_map` ONCE before the episode
  loop -> stale on fresh_map (map redrawn per episode) -> G7 FAIL 0.27 (spurious).
  Fixed to read obs_map per-episode after generate_trajectory. Verified oracle=1.0
  within-episode manually. Task is valid; the n-gram gate (load-bearing) passed.
- run_mw_fresh_factorial.sh gate-abort greps only G4; SHOULD also check G7 next time
  (harmless here since G7 was a validator bug, but make the gate check both).

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
