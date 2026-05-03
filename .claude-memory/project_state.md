---
name: MapFormer project state
description: Current empirical state of the MapFormer reproduction + extensions. Use to orient on resume; verify against git log + RESULTS_PAPER.md before recommending.
type: project
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
Repo: /home/prashr/mapformer (single-author project; pushes to GitHub PrashRangarajan/mapformer).

**Headline result (multi-seed, n=3):** Level 1.5 InEKF on MapFormer-WM beats Vanilla MapFormer-WM by ~11pp on noise/landmark OOD T=512, with calibrated uncertainty (NLL 2× lower). Source of truth: `RESULTS_PAPER.md`.

**Reframings (important, supersede earlier docs):**
- Kalman's win is **stabilisation + token-type gating**, not Bayesian inference. The wrap (atan2 of innovation) keeps θ̂ bounded at OOD length; per-token R_t gates by token type. R_t saturates HIGH on aliased obs yet Level 1.5 still beats Vanilla — the gain is structural, not inferential.
- PC and InEKF are **mathematical duals**, not complements. Coupling them creates a degenerate optimum (R-saturation autoencoder bypass).
- Action-record corruption (our `--p-action-noise`) is **mathematically equivalent to a stochastic-transition MDP** for uniform policies. Use the stochastic-transition framing — it's standard control/RL vocabulary. `--p-transition-noise` flag added 2026-05-01 for direct empirical validation.
- **NLL is the better metric** than accuracy for comparing these architectures. Two models can be tied on accuracy with NLL differing by 2×. Level 1.5 dominates NLL across all regimes.

**Important bug fix (2026-05-01):** Earlier `model_level15_dog.py` and the new `continuous_nav.py` computed DoG targets as `max(0, exp(-d²/2σE²) - exp(-d²/2σI²))` which gives **all-zero targets** (both Gaussians = 1 at d=0). The earlier `DOG_RESULTS.md` (max grid score 0.036) was on broken targets — VACUOUS, not a real test of Sorscher's conditions. Fixed to use normalised 2D Gaussians (1/σ² prefactor); supersedes `DOG_RESULTS.md` with `DOG_RESULTS_FIXED.md` (re-run pending GPU availability).

**Continuous 2D nav infrastructure (NEW 2026-05-01):**
- `continuous_nav.py`: ContinuousNav2D + ContinuousNav2D_Cached. SE(2)-flavour state on a torus, velocity commands with explicit Gaussian process noise, DoG-of-position obs targets.
- `model_continuous.py`: MapFormerWM_Continuous + MapFormerWM_Continuous_Level15 with optional ReLU bottleneck (`n_grid_units > 0`) for hex probing.
- `train_continuous.py`: separate trainer for continuous task (MSE on action-position predictions, not next-token CE).
- `probe_hex_continuous.py` + `eval_continuous.py`: hex probe + cross-T cross-noise eval. Currently in flight.

**MiniGrid infrastructure (added 2026-04-30/05-01):**
- `minigrid_env.py`: MiniGridWorld + MiniGridWorld_Cached (~35× speedup via 25K-trajectory pre-built buffer). MiniGrid envs registered in train_variant.py: minigrid_empty, minigrid_doorkey, minigrid_doorkey16, minigrid_multiroom, minigrid_keycorridor, minigrid_obstructedmaze.
- DoorKey-8x8 results: Level15 wins +10pp at OOD T=512 noise, ties on clean (small env, drift sub-cell).
- DoorKey-16x16 in `MINIGRID_DK16_RESULTS.md`.
- Long-T (up to T=2048) eval in `MINIGRID_DOORKEY_LONGT.md` shows: clean accuracy converges, Level15's NLL stays lower; noise accuracy gap *grows* with T (+16pp at T=2048).
- RoPE diagnostic in `MINIGRID_DOORKEY_ROPE_DIAG.md`: RoPE collapses at long T; MapFormer/Level15 hold. Validates that the env exercises path integration.
- IMPORTANT cached-vs-live discrepancy: cached buffer reuses 25K trajectories across all epochs; live regenerates each batch. Numbers differ subtly on clean (Level15 loses cached, wins live by ~2pp). Trajectory diversity matters; note when reporting cached results.

**v4 control resolution (closed 2026-04-30):** v4's modest +3pp lm200 win does NOT come from RNG drift (Level15 with `forward_model` instantiated-but-untrained gives bit-identical numbers to Level15 alone). The +3pp is from grad-clip coupling: forward_model gets nonzero gradient from aux loss, the joint norm includes its contribution, and clip rescales the main-model gradient. Surprising side effect, not a PC mechanism.

**Variants registered in `train_variant.py`:** Vanilla, VanillaEM, RoPE, Level1, Level15, Level15EM, Level15EM_b5, Level2, PC, Level15PC, Level15PC_NoBypass, Level15PC_v3, Level15PC_v4, Level15_DoG, Grid, Grid_Free, GridL15PC, GridL15PC_Free, ablations (L15_ConstR/NoMeas/NoCorr/DARE), baselines (LSTM, CoPE, MambaLike).

**Pipelines that ran (2026-05-01 → 2026-05-02):**
- P1 ✅ `DOG_RESULTS_FIXED.md`: hex still doesn't emerge on discrete torus even with the fixed DoG kernel. Max grid score 0.042; 0% units > 0.3. Likely needs continuous-state, not discrete-cell, navigation.
- P2 ❌ `CNAV_RESULTS.md`: training collapsed because MSE loss has degenerate near-zero minimum on sparse DoG targets. All 4 variants got ~20-cell mean position error (chance). Fix: replaced with `hard_ce` loss in `train_continuous.py`; smoke-test goes from chance to ~2-cell error in 500 steps. Re-run pipeline `run_cnav_redo.sh` polling for free GPUs.
- P3 ✅ `STOCHASTIC_TRANSITION_RESULTS.md`: action-record corruption ≡ stochastic-transition MDP for uniform policy, with small (~5pp) on/off-diagonal asymmetry. trans-noise training generalizes slightly better than action-noise training because trajectory diversity is higher.
- P4 ✅ `MINIGRID_MEMORY_RESULTS.md` (after kwarg-bug fix): **Level15 wins decisively on MemoryS13** — +13pp over Vanilla at T=512 OOD, +13pp at T=1024, NLL 5× better. RoPE collapses faster than Vanilla. The rooms+hallway topology genuinely tests path integration. This is the cleanest "Level15 wins clean OOD on a real env" result we have.
- P5 ✅ `TEM_RESULTS.md` (after orthogonal-W_a fix): **TEMFaithful is the worst baseline** — much worse than TEM-Lite (GRU) and ~50pp behind Vanilla MapFormer on clean OOD T=512 (0.423 vs 0.862). Likely too restrictive at d_g=64 + single-env training; matched-capacity ablations needed.

**Important bug fixes during 2026-05-02:**
- `minigrid_env.py`: MiniGridWorld_Cached now accepts (and ignores) `p_transition_noise` kwarg (was crashing).
- `model_tem_faithful.py`: per-action W_a now parameterised as `exp(skew(A_a))` — orthogonal by construction, no NaN. Matches Whittington 2019's compact-group convention.
- `train_continuous.py`: replaced MSE default with `hard_ce` for CNAV. MSE has degenerate zero-minimum on sparse DoG targets.

**Not yet done / open:**
- Re-run CNAV with hard_ce: `run_cnav_redo.sh` polling for free GPUs as of 2026-05-02.
- TEM-style multi-environment training as alternative hex route (untested).
- MAmPa baseline (paper's own block-diagonal Mamba variant).
- Refresh `paper_figures/` to include VanillaEM, Level15EM, MambaLike, LSTM, RoPE rows.
- Full SE(2) InEKF (currently only SO(2)/rotation correction; translation correction is a follow-up).
- **TEM-t baseline** (transformer formulation of TEM): the *parameter-matched* comparison the paper claims to make. Sequential per-action W_a + transformer attention scaffolding — same total params as MapFormer-EM (~250K), only differs in sequential-W_a vs parallel-cumsum-f_Δ. The clean test of the parallelism-vs-expressivity claim. **Suggested by user 2026-05-02 because TEMFaithful's 19K vs MapFormer's 250K isn't an apples-to-apples comparison.**
- **TEMFaithful matched-capacity ablations** (after the TEM-t comparison): TEM-Big scaled to ~250K total params, MapFormer-Tiny stripped to ~19K, TEM-LargeWa with d_g=256.

**Pipeline conventions:** Torus training ~10s/epoch on GPU. MiniGrid live ~360s/epoch (gym.step bottleneck); cached ~1.7s/epoch. Continuous nav: similar to torus once buffer is built. Multi-seed = 3 seeds. Shell scripts auto-train + eval + git commit + git push. Memory writes go to `.claude-memory/` (in repo, synced via git).

**How to apply:** When the user asks "what's the state?" consult `SESSION_2026-05-01.md` for the most recent discussion thread and reframings, `RESULTS_PAPER.md` for the canonical results table, and the latest commit log for what's actually landed. When recommending a new experiment, check `train_variant.py::VARIANT_MAP` for available variants and confirm any in-flight pipelines (look for `run_*.sh` processes) before launching new ones.
