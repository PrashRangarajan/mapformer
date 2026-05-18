# Addendum to REPORT.md — work since 2026-05-15

Extends the consolidated `REPORT.md` (commit `c21981d`) with experiments
landed afterward. Same source-of-truth conventions: every claim cites its
MD; multi-seed where noted.

---

## 1. Active-inference one-step probe — **NEGATIVE**

Source: `ACTIVE_INFERENCE_RESULTS.md`.

**Setup:** Random-walk-trained single-env lm200 checkpoints. No goal
conditioning, no policy training. At each navigate step, for each
candidate action `a`, score by `max log p(goal_token)` over a `horizon`-step
rollout (Dreamer-style argmax-obs unfolding for horizon > 1). Pick argmax.
Closed-loop execution. T_explore=64, T_navigate=32, 50 episodes, s0.

**Result:**

| Variant | h=1 | h=4 | h=8 |
|---|---|---|---|
| RoPE | 0.00 | 0.00 | 0.00 |
| Vanilla | 0.00 | 0.02 | 0.00 |
| Level15 | 0.02 | 0.02 | 0.00 |
| TEMFaithful | 0.02 | 0.00 | 0.00 |

**Verdict:** 0–2% across all variants and horizons. The trained forward
model is structurally too myopic — `p(goal_token | seq + a)` is only
informative when the agent is adjacent to the goal. Multi-step argmax-obs
rollouts rarely pass through the goal cell. Cannot drive multi-step
navigation from a next-token-only model via Friston-style action selection.
**Confirms representational story is intact but behaviour requires more
than the trained forward model + a clever decision rule.**

---

## 2. Single-size Level15 control — coupled-ω hypothesis **CONFIRMED partially**

Source: `SINGLE_SIZE_CONTROL.md`.

**Setup:** Train Level15 lm200 on a SINGLE grid size (no multi-scale
mixing). Compare to existing `Level15_multisize` (which trained on a mix
of 32/64/128 with shared ω). If single-size closes the gap to TEM at
size 32, the coupled-ω was the bottleneck.

| Variant | size 16 T=512 | size 32 T=512 |
|---|---|---|
| **Level15 single-size** | 0.159 ± 0.019 | **0.908 ± 0.085** |
| Level15 multi-size (32+64+128) | — | 0.782 ± 0.138 |
| TEMFaithful multi-size (reference) | — | 0.936 ± 0.021 |

**Single-size 32 = 0.908 vs multi-size 32 = 0.782 → +13pp**. Closes most
of the gap to TEM (0.936). **The coupled-ω hypothesis is empirically
confirmed.**

Size 16 collapses for all (the architecture's `n_blocks = 32` has more
frequency channels than the 256-cell grid has positions — different
failure mode, not informative for this question).

---

## 3. Per-scale ω architectural fix — **PARTIAL win**

Source: `PERSCALE_OMEGA_RESULTS.md`. Multi-seed n=3.

**Setup:** `MapFormerWM_Level15_PerScaleOmega` — one learnable ω per
training scale, selected at forward time by an `env_sizes` kwarg. Same as
Level15 otherwise. Trained on the same multi-scale mix.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| Level15 (coupled ω) | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| **Level15_PerScaleOmega** | **0.877 ± 0.163** | 0.937 ± 0.075 | 0.946 ± 0.074 |
| TEMFaithful (reference) | 0.936 ± 0.021 | 0.973 ± 0.006 | 0.981 ± 0.005 |

**Size 32: +10pp over coupled-ω Level15 (0.782 → 0.877).** Closes about
half the gap to TEM. Size 64/128 essentially tied (expected null —
coupled-ω wasn't hurting those scales).

**Verdict:** Per-scale ω is a real architectural improvement at small
grids, but it explains only ~half of TEM's small-grid advantage. There's
a second factor (likely TEM's scale-agnostic Hopfield retrieval + W_a
parametrisation) still in play. High variance (±0.163) — at least one
seed under-converged. Workshop-honest framing: "we identify and partly
fix the coupled-ω limitation; full parity at small grids needs further
work."

---

## 4. Successor-representation aux pretraining — **NEGATIVE**

Source: `SR_PRETRAIN_RESULTS.md`.

**Setup:** `MapFormerWM_Level15_SR` — Level15 + auxiliary head
`Linear(d_model, vocab_size)` trained with BCE-with-logits to predict
"token v appears in positions t+1..t+K=8" at every position. Joint with
standard next-token CE. Coefficient 0.5. Trained on single-env lm200.

Then re-ran the two probes that failed on Level15 baseline (goal-distance,
active-inference) to test whether the supervised multi-step-reachability
signal lifts them above chance.

**Training:** Final CE loss 0.016 ± 0.007 (n=3). Pretraining converged
fine.

**Goal-distance probe (s0):**

| Variant | heldout MAE / Spearman |
|---|---|
| Level15 (no SR) | 14.35 / -0.02 |
| Level15_SR | 14.86 / **-0.27** |

**Active-inference closed-loop (s0):**

| Variant | horizon=1 | horizon=4 |
|---|---|---|
| Level15 (no SR) | 0.02 / 0.02 | 0.02 / 0.02 |
| Level15_SR | **0.00** | **0.00** |

**Verdict: complete null.** Worse than baseline on both probes despite
training converging cleanly.

**Why it likely failed:** the aux head is a parallel function. The linear
SR head can predict "token v in next K" *without* forcing the backbone's
hidden state to encode multi-step reachability differently — the head
extracts future-token info from the same representation that's already
optimized for next-token CE. To force the backbone to change, we'd need
either:
- Tying the SR head to the same weights as out_proj (no separate parameters).
- Goal-conditional supervision — train on (goal_token, trajectory) pairs
  with a goal-relative target.
- True model-based RL with the cognitive map as the world model.

**Implication:** aux losses on the same backbone with a separate head are
insufficient to bridge representation → behaviour. The "additional
training signal" lesson is that the signal needs to constrain the BACKBONE
itself, not just give a parallel decoder.

---

## 5. Engineering fix: recurring cwd bug in run_*.sh aggregators — **CLOSED**

Source: `feedback_cwd_aggregator_bug.md`.

Three times in 2026-05-14/05-15 sessions, aggregator MDs came back with
every row blank because:
1. Run scripts `cd /home/prashr` at the top (so `python3 -m mapformer.X`
   resolves).
2. Python heredocs with relative paths (`runs/X`, `paper_figures/Y`) then
   resolve to `/home/prashr/...` not `/home/prashr/mapformer/...`.

Patched 7 affected scripts to `cd "$REPO"` immediately before each Python
heredoc:
- `run_level15_novelenv_fillin.sh`
- `run_post_pipeline_analysis.sh`
- `run_tem_novel_envs.sh`
- `run_tem_background_baselines.sh`
- `run_multienv_clean_baselines.sh`
- `run_multiclass_multiseed.sh`
- `run_vanilla_nodrop_control.sh`

Memory rule saved so future chats default to the right pattern. Audit
command: `grep -B1 "python3 -u <<" run_*.sh | grep -v "cd \"\$REPO\"" | grep "python3"`.

---

## 6. Updated workshop pitch state

Net change since `REPORT.md`:

**Representational story is stronger** — `TEM_NOVEL_ENV_RESULTS.md` is now
fully populated across 4 axes (Level15 cross-topology + cross-scale rows
landed via `run_level15_novelenv_fillin.sh`). Defensible workshop one-line:

> "MapFormer extends to novel environments along four axes — held-out
> same-class, cross-topology, cross-scale, cross-class. Across all four,
> Level1.5 correction matches or exceeds TEM-style explicit memory."

**Behavioural story is conclusively closed for the workshop** — we've
exhausted the cheap planning-via-frozen-model options:
- Closed-loop BC: 1-2% (BC distribution shift)
- DAgger: 0.42 ceiling for Level15NoDrop only
- Goal-distance probe (head state): chance
- Goal-distance probe (explicit state): half-chance on displacement
- Active inference one-step / multi-step: 0-2%
- **SR-aux pretraining: chance** ← new this session

Honest framing for the workshop pitch:

> "We characterise which transformer cognitive-map architectures build
> usable spatial representations for OOD generalization. Connecting
> these representations to closed-loop goal-directed behavior remains
> an open problem; the cheap bridges (BC, DAgger, frozen-model planning,
> SR-aux pretraining) all fall short of the representational ceiling."

**New honest-positive results to fold into headline:**
- Per-scale ω is a +10pp architectural improvement at small grids. New row in `PERSCALE_OMEGA_RESULTS.md`.
- VanillaNoDrop control proves InEKF correction is doing real work (committed in REPORT.md but referenced for context).

**Negative results worth flagging in caveats:**
- Active inference doesn't work as a frozen-model decision rule.
- SR aux pretraining doesn't bridge the representation-behaviour gap.
- Per-scale ω only partly closes the small-grid gap.

---

## Appendix — new source files (this session)

Variants:
- `model_vanilla_nodrop.py` (`VanillaNoDrop`) — control for InEKF necessity
- `model_inekf_level15_perscale.py` (`Level15_PerScaleOmega`) — per-scale ω
- `model_inekf_level15_sr.py` (`Level15_SR`) — SR-aux head

Probes:
- `probe_active_inference.py` — one-step + multi-step EFE-style planning
- `probe_position_decode.py` — displacement-from-start sanity check
- `probe_goal_distance_state.py` — explicit-state version of goal-distance

Eval:
- `eval_single_env.py` — generic single-env eval used to fill missing rows
- `make_place_cell_figure.py` v2 — TEM-g aware

Run scripts (auto-pipelines): `run_planning_and_scale_controls.sh`,
`run_perscale_omega.sh`, `run_sr_pretrain_and_probe.sh`, `run_fixup.sh`,
`run_state_probes.sh`, `run_post_pipeline_analysis.sh`,
`run_level15_novelenv_fillin.sh`, `run_tem_novel_envs.sh`,
`run_tem_background_baselines.sh`, `run_multienv_clean_baselines.sh`,
`run_multiclass_multiseed.sh`, `run_vanilla_nodrop_control.sh`.

Memory:
- `feedback_cwd_aggregator_bug.md` (new)
- `feedback_seed_ordering.md` (new)
- `feedback_minimal_sweep_skip_gsf.md` (new)
- `project_state.md` (refreshed for cross-session handoff)

*Generated 2026-05-18. See REPORT.md for the cumulative table; this file
contains only deltas since then.*
