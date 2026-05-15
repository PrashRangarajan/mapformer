---
name: MapFormer project state
description: Current empirical state of the MapFormer extensions, in-flight pipelines, and the workshop pitch. Cross-session handoff. Verify against git log before recommending.
type: project
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---

Repo: `/home/prashr/mapformer` — single-author, pushes to GitHub `PrashRangarajan/mapformer`. Cross-session memory at `.claude-memory/` (synced via repo).

## Currently in flight (as of 2026-05-15 evening)

**`run_level15_novelenv_fillin.sh`** (PID 1384369) — fills two remaining gaps in `TEM_NOVEL_ENV_RESULTS.md`:
- Level15 (single, no GSF) on cross-topology, n=3
- Level15 (single, no GSF) on cross-scale, n=3
- Seed-outer-loop ordering. Currently in seed 1 of 3 as of last check; ETA ~1h remaining.
- Auto-regenerates `TEM_NOVEL_ENV_RESULTS.md` (4 sections: multi-env / cross-topology / cross-scale / cross-class) and auto-commits + pushes when done.

Once it finishes, **the "Can MapFormer do TEM's novel-env setting?" question is fully answered** — apples-to-apples comparison across RoPE / Vanilla / Level15 / Level15GSF_NoDrop[_K16] / TEMFaithful on all four novel-env axes. The only remaining n/a cell is TEMFaithful cross-class (vocab structure incompatibility — `tokens<n_actions` doesn't fit the unified vocab; would require a custom is-action mask).

## Headline empirical state — TEM-setting novel envs

**The defensible workshop one-line claim:**
> "MapFormer extends to novel environments along four axes — held-out same-class, cross-topology, cross-scale, cross-class. Across all four, our Level1.5 correction matches or exceeds TEM-style explicit memory."

Multi-seed (n=3) numbers, all in repo MDs:

- **Multi-env held-out LM200 T=512 OOD** (`TEM_NOVEL_ENV_RESULTS.md`, `MULTIENV_CLEAN_2x2.md`):
  RoPE 0.503, Vanilla 0.728, Level15 0.988, Level15GSF_NoDrop 0.976, TEMFaithful 0.967.
- **Multi-env held-out CLEAN T=512 OOD**: RoPE 0.503, Vanilla 0.920, Level15 0.975, Level15GSF_NoDrop 0.989, TEMFaithful 0.976.
- **Cross-topology T=512** (torus / open / walls): Level15GSF_NoDrop_K16 0.955 / 0.855 / 0.778; TEMFaithful 0.907 / 0.823 / 0.796. Level15 row landing now.
- **Cross-scale T=512** (size 32 / 64 / 128): Level15GSF_NoDrop_K16 0.826 / 0.951 / 0.973; **TEMFaithful 0.936 / 0.973 / 0.981** (TEM dominates small grids — see `TEM_CROSSSCALE_DIAGNOSTIC.md` for the coupled-ω hypothesis). Level15 row landing now.
- **Cross-class** (torus + DoorKey, `MULTICLASS_MULTISEED_RESULTS.md`): Level15 torus 0.879 / DoorKey 0.888; Level15GSF_NoDrop_K16 0.865 / 0.891. RoPE 0.497 / 0.788, Vanilla 0.681 / 0.841. TEMFaithful n/a (vocab).

## Critical controls (already landed)

- **VanillaNoDrop control** (`VANILLANODROP_CONTROL.md`, n=3): VanillaNoDrop ≈ Vanilla on lm200 (0.737 vs 0.728) and on clean (0.962 vs 0.920). **InEKF correction is doing real work; the Level15NoDrop +13pp lm200 win is NOT reducible to a dropout bug.** This is the workshop-critical control.
- **Multi-env clean 2×2 disambiguation** (`MULTIENV_CLEAN_2x2.md`): Vanilla on multi-env clean recovers to 0.920 (vs single-env lm200 0.72), confirming multi-env training generally helps attention. But on multi-env LM200 it stays at 0.728 — **landmarks specifically still demand architectural correction.** Cognitive-tier all in 0.97-0.99 either way.

## NoDrop vs GSF (the substitutes-not-complements finding)

From `GSF_NODROP_RESULTS.md` (n=3 lm200 OOD T=512):
| | acc | NLL |
|---|---|---|
| Level15 | 0.819 | 0.897 |
| Level15NoDrop | 0.948 | 0.317 |
| Level15GSF | 0.956 | 0.227 |
| Level15GSF_NoDrop | 0.961 | **0.177** |
| TEMFaithful | 0.969 | 0.171 |

NoDrop and GSF are **accuracy-substitutes** (+13pp vs +14pp; stacked still +14pp), but **NLL-complements** (stacked 5× better NLL). Workshop-relevant framing: NoDrop is free engineering; GSF is the principled choice for calibrated cognitive maps for downstream planning.

## Place cells

`paper_figures/place_cells_per_variant.png` (committed `691762a`). After fixes (probe TEM's `g` not `out_proj`; smoothing σ=1.5; visit mask ≥5):
- TEMFaithful: peak ratios 9447× / 6349× / 317× — by far cleanest
- Level15: 88×, Level15GSF_NoDrop: 97×, RoPE: 424×, Vanilla: 37×

The previous "TEM looks worst" was an artifact of probing `LN(x_hat)` (content readout) instead of `g`. Hex emerges in zero variants (consistent with Sorscher).

## Closed-loop & goal-conditioning (still problematic)

- **Goal-conditioned closed-loop on torus** (`GOAL_CLOSEDLOOP_RESULTS.md`, single seed): every variant succeeds at 1-2% closed-loop despite match-acc 0.92-0.95. BC distribution shift dominates. Can NOT use this as a workshop claim as currently structured.
- **DAgger on DoorKey-8x8** (`DAGGER_RESULTS.md`): Level15NoDrop 0.24 → 0.42 (+18pp) is the one clear DAgger win; Vanilla and Level15EM degrade. NoDrop matters more under DAgger because post-attn features carry recovery patterns.

**Goal-distance probe** (`PROBE_GOAL_DISTANCE.md`): all variants at chance MAE; only TEMFaithful shows weak Spearman (0.27). The post-LN hidden state does NOT contain linearly-decodable goal-relative distance.

**State-level probe** (`STATE_PROBES.md`, s0): probing explicit spatial state (`theta_hat` for Level15, `theta_path` for Vanilla, `g` for TEM) with displacement-from-start as target:
- Vanilla: 16.05 cells MAE (chance 32.15)
- Level15: 15.32 cells MAE (chance 32.08)
- TEMFaithful: similar order

The cognitive map IS there in `theta_hat` — about half-chance error on displacement decoding — but not super sharp. Still ~15 cells out on a 64×64 torus.

## Workshop pitch direction (discussed but undecided)

Two threads:
1. **"Why we care about cognitive maps"**: representational generalization across env axes. Strongest current evidence; what the in-flight fillin pipeline completes.
2. **"What is the cognitive map for"**: behavioral / planning side. Currently weak — closed-loop fails for all, goal-distance probe near chance. The literature anchor is active inference / predictive coding / world models (Friston, Da Costa, Hafner Dreamer, MuZero). Recommendation: pitch as "characterize which cognitive-map architectures meet the *usable-internal-model* assumption that planning literature makes." Three concrete protocols sketched in earlier session text — most promising is `(3)` goal-distance probe, which already partly ran.

Bigger-models / workshop scale (4 layers, d=256, ~2-4M params) discussed but NOT yet started.

## Key files added 2026-05-15 session

- `model_vanilla_nodrop.py` — `MapFormerWM_VanillaNoDrop`, registered in train_variant.py
- `probe_goal_distance.py` — goal-distance probe (head-state version). Bug fixed (use `model_emb.embedding_dim` not `cfg["d_model"]` for d_goal).
- `probe_goal_distance_state.py` — same probe but reads `theta_hat` / `theta_path` / `g` explicitly.
- `probe_position_decode.py` — sanity check: does explicit spatial state encode displacement-from-start `(dx, dy)`?
- `eval_single_env.py` — generic single-env eval (used to fill TEM single-env clean row).
- `make_place_cell_figure.py` v2 — probes TEM's `g`, applies smoothing + visit mask.
- Run pipelines: `run_tem_novel_envs.sh`, `run_tem_background_baselines.sh`, `run_multienv_clean_baselines.sh`, `run_vanilla_nodrop_control.sh`, `run_multiclass_multiseed.sh`, `run_post_pipeline_analysis.sh`, `run_fixup.sh`, `run_state_probes.sh`, `run_level15_novelenv_fillin.sh`.

## Naming + ordering conventions reinforced this session

- **Seed-outer-loop:** `for seed in [0,1,2]: for variant in [...]: train()`. Lands full-coverage low-confidence table fastest; early failure detection. See `feedback_seed_ordering.md`.
- **Skip GSF in minimal sweeps:** accuracy-redundant with NoDrop at ~4× compute. See `feedback_minimal_sweep_skip_gsf.md`.
- **Cwd bug warning:** scripts that `cd /home/prashr` then run Python heredocs with relative `paper_figures/...` paths resolve to `/home/prashr/paper_figures/...`, NOT `/home/prashr/mapformer/paper_figures/...`. Aggregators must `cd $REPO` before running, or use absolute paths.

## Known structural blockers / honest n/a

- **TEMFaithful cross-class**: `tokens<n_actions` action/obs split doesn't fit the unified vocab (env-prefix tokens 0/1 + 11 actions over IDs 2-12). Workshop table can flag as "n/a, vocab incompatible."
- **Closed-loop / DAgger ceiling** on DoorKey: ~0.20-0.25 across variants. Architectural differences don't cleanly translate to closed-loop behaviour — match-acc is a lossy proxy.

## How to pick up next session

1. **First action:** check `git log -5` and `ps -ef | grep run_` to see if the fillin pipeline finished. If yes, `TEM_NOVEL_ENV_RESULTS.md` will have all 4 sections fully populated.
2. **Workshop next concrete steps:** either (a) scale up Level15 to 4-layer d=256 for the workshop-scale claim, or (b) build the active-inference / world-model planning protocol that connects to the literature. (b) is more differentiated; (a) is more standard. User leaned toward both — "needs to be done on some kind of bigger model. Also, we need to nail down why we care about having a cognitive map."
3. **Open question still on the table:** TEM cross-scale dominance — `TEM_CROSSSCALE_DIAGNOSTIC.md` proposes per-scale ω head for Level15 as the falsifiable test. Cheap and would either close the gap or confirm the limitation.
