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

**Session 2026-05-10 — TEMFaithful fix, Level15Beta + dropout discovery, EM/WM mechanism, goal-directed task:**

- **TEMFaithful predict-then-update bug fix.** Old TEMFaithful queried memory with PRE-action g (wrong cell's content). Fix: update g via W_a BEFORE prediction. Took lm200 OOD T=512 from 0.42 (chance) to **0.969**. The "TEMFaithful is the worst baseline" finding in the prior session was driven by this bug; reverse the framing — TEMFaithful is now the lm200 leader.
- **TEM-t NaN fix.** Unconstrained ReLU(e·W_a) recurrence → ||e|| explodes ~10× per 8 steps → 1e13 by L=255 → NaN. Fixed with two LayerNorms: `e_pre_attn` (paper-faithful) and `e_in_rnn` (deviation; replaces paper's sensory-landmark reset which we lack).
- **Level15Beta (learnable softmax temperature β) + dropout discovery.** Beta closed +12pp lm200 gap (0.819 → 0.935). Learned β barely moved (0.148–0.182 vs init 0.125), so β cannot explain it. Level15NoDrop (fixed β, only post-attn residual dropout removed) gives **0.948** — confirms the win was dropout removal. β was a red herring. See `feedback_post_attn_dropout.md`. Regime-dependent Pareto trade-off: dropout removal helps lm200 +12pp, helps noise +2pp, mildly hurts clean (NLL doubles).
- **EM-vs-WM mechanism (paper-citable).** EM's `A = softmax(A_X ⊙ A_P)` is a multiplicative AND-gate; WM's additive scoring is an OR-gate. EM wins when A_X is the noisy channel (paper's tasks: aliased obs, especially large vocab at l=16). WM wins when A_X is the signal channel (our tasks: landmarks, long OOD, noise). Backbone choice is regime-dependent. See `feedback_em_vs_wm_mechanism.md`.
- **MapFormer paper scaling claims (verified via WebFetch).** Figure 4: EM > WM along (a) head size 16→128 at l=256, (b) sequence length 16→384 at h=48, (c) vocab 10→10000 at l=16. All compatible with the mechanism above. None test our extended regimes (long OOD with rare landmarks).
- **New model files:** `model_inekf_level15_beta.py` (β layer), `model_inekf_level15_nodrop.py` (clean dropout ablation), `model_inekf_gsf.py` (Gaussian Sum Filter, K parallel Level 1.5 chains — registered as `Level15GSF`, smoke-tested, **not yet trained**).
- **Goal-directed navigation task (new infra):** `environment_goal.py` (GoalDirectedGridWorld + BFS-on-torus oracle), `train_goal.py` (CE on next-action prediction at navigate-phase positions). Episode = `[goal_token, explore_phase_random_walk, navigate_phase_bfs]`. Goal token = the landmark's unified-vocab emit. Smoke test: 3 epochs → 0.708 held-out action accuracy (chance 0.25).

**Vocab sweep results (2026-05-10):** `VOCAB_SWEEP_RESULTS.md` (single seed, clean task, T=128 train / T=512 OOD on fresh obs_map).
- n_obs=16 (paper main): Vanilla 0.862, VanillaEM 0.968, Level15 0.991, Level15EM 0.986. EM-with-correction nearly tied with WM-with-correction.
- n_obs=256: Vanilla 0.665, VanillaEM 0.562 (EM WORSE), Level15 0.980, Level15EM 0.970. Paper's "EM wins at large vocab" claim does NOT survive at our l=128/T=512. Vanilla EM collapses (vocab outpaces capacity); correction rescues both backbones.
- n_obs=4096: ALL variants collapse to ~0.45 — degenerate regime (each cell emits a near-unique token, test-env obs_map is completely different). Uninformative.
- Verdict: at long-l regime, vocab scaling does NOT invert the WM-EM ordering. Paper's Fig 4c result is l=16-specific. Backbone matters less than correction at long l.

**Goal-directed navigation results (2026-05-10):** `GOAL_DIRECTED_RESULTS.md` (single seed, lm200, 50 epochs).
- Vanilla: 0.628 / 0.950 / 0.766 (T_exp=32 / 64=train / 128=OOD). **Vanilla cognitive maps degrade with longer explore — drift accumulates → action selection breaks.**
- Level15: 0.939 / 0.947 / 0.950. **Correction-stabilised maps stay navigable across all explore lengths** — this is the bounded-error Kalman promise made concrete on a goal-directed task.
- Level15EM: 0.936 / 0.949 / 0.948. Tied with Level15 — the multiplicative AND-gate is benign when correction repairs A_P enough that the gate fires reliably.
- Level15NoDrop: 0.939 / 0.946 / 0.949. Dropout removal has NO effect on goal-directed task (which is 4-class action prediction — retrieval is dense, not rare).
- **Best finding for paper:** cognitive maps built with Level 1.5 correction are GOAL-NAVIGABLE and STAY navigable under OOD explore length. Vanilla maps degrade. This is the cleanest cognitive-map utility test we've done — connects path-integration correction to behaviour.

**Pending decisions:**
- GSF launch: depends on vocab sweep + goal-directed results. The dropout finding weakens the "multimodal Bayes is the missing piece" story — GSF would now mostly test compute-vs-K scaling rather than "does K=8 Kalmans match TEMFaithful." Worth running but reframe expectations.
- Level15NoDrop multi-seed on clean + noise (currently only lm200 has multi-seed; needed to nail down the Pareto trade-off).

**Older not-yet-done / open:**
- Re-run CNAV with hard_ce: `run_cnav_redo.sh` polling for free GPUs as of 2026-05-02.
- TEM-style multi-environment training as alternative hex route (untested).
- MAmPa baseline (paper's own block-diagonal Mamba variant).
- Refresh `paper_figures/` to include VanillaEM, Level15EM, MambaLike, LSTM, RoPE rows.
- Full SE(2) InEKF (currently only SO(2)/rotation correction; translation correction is a follow-up).
- **TEM-t baseline** (transformer formulation of TEM): the *parameter-matched* comparison the paper claims to make. Sequential per-action W_a + transformer attention scaffolding — same total params as MapFormer-EM (~250K), only differs in sequential-W_a vs parallel-cumsum-f_Δ. The clean test of the parallelism-vs-expressivity claim. **Suggested by user 2026-05-02 because TEMFaithful's 19K vs MapFormer's 250K isn't an apples-to-apples comparison.**
- **TEMFaithful matched-capacity ablations** (after the TEM-t comparison): TEM-Big scaled to ~250K total params, MapFormer-Tiny stripped to ~19K, TEM-LargeWa with d_g=256.

**Pipeline conventions:** Torus training ~10s/epoch on GPU. MiniGrid live ~360s/epoch (gym.step bottleneck); cached ~1.7s/epoch. Continuous nav: similar to torus once buffer is built. Multi-seed = 3 seeds. Shell scripts auto-train + eval + git commit + git push. Memory writes go to `.claude-memory/` (in repo, synced via git).

**How to apply:** When the user asks "what's the state?" consult `SESSION_2026-05-01.md` for the most recent discussion thread and reframings, `RESULTS_PAPER.md` for the canonical results table, and the latest commit log for what's actually landed. When recommending a new experiment, check `train_variant.py::VARIANT_MAP` for available variants and confirm any in-flight pipelines (look for `run_*.sh` processes) before launching new ones.
