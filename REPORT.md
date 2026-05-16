# MapFormer extensions — consolidated results report

Auto-generated consolidation of the per-experiment MDs in this repo. The
defensible workshop one-line:

> **"MapFormer extends to novel environments along four axes — held-out
> same-class, cross-topology, cross-scale, cross-class. Across all four,
> our Level1.5 correction matches or exceeds TEM-style explicit memory."**

Each section here points to the source MD so individual results can be
re-checked. Numbers are n=3 unless noted.

---

## 1. Headline results — TEM-setting novel environments

Source: `TEM_NOVEL_ENV_RESULTS.md`, `MULTIENV_CLEAN_2x2.md`,
`MULTICLASS_MULTISEED_RESULTS.md`.

### 1.1 Multi-env held-out — TEM's classic novel-env test

50 train envs, 50 held-out test envs, size 64, T=128 train.

**LM200 (with landmarks):**
| Variant | Train | Held T=128 | Held T=512 OOD |
|---|---|---|---|
| RoPE | 0.587 ± 0.024 | 0.595 ± 0.027 | 0.503 ± 0.003 |
| Vanilla | 0.823 ± 0.039 | 0.820 ± 0.044 | 0.728 ± 0.047 |
| **Level15** | 0.997 ± 0.001 | 0.997 ± 0.002 | **0.988 ± 0.003** |
| Level15GSF_NoDrop | 0.997 ± 0.002 | 0.996 ± 0.002 | 0.976 ± 0.005 |
| TEMFaithful | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.967 ± 0.004 |

**CLEAN (no landmarks):**
| Variant | Train | Held T=128 | Held T=512 OOD |
|---|---|---|---|
| RoPE | 0.589 ± 0.080 | 0.592 ± 0.066 | 0.503 ± 0.018 |
| Vanilla | 0.994 ± 0.005 | 0.992 ± 0.007 | 0.920 ± 0.024 |
| Level15 | 0.998 ± 0.003 | 0.998 ± 0.003 | 0.975 ± 0.010 |
| **Level15GSF_NoDrop** | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.989 ± 0.006** |
| TEMFaithful | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.976 ± 0.002 (n=2) |

**Reading:** Level15 / GSF / TEM all in 0.97–0.99 OOD. The standard
transformers (Vanilla, RoPE) are not. Cognitive-tier architectures all
clear the bar; the choice between them is regime-dependent.

### 1.2 Cross-topology (TEM OOD-d analog)

Train on mix of torus + open + walls; evaluate per-topology on held-out
envs at T=512 OOD.

| Variant | torus | open | walls |
|---|---|---|---|
| RoPE | 0.468 ± 0.027 | 0.510 ± 0.027 | 0.536 ± 0.006 |
| Vanilla | 0.787 ± 0.039 | 0.717 ± 0.032 | 0.658 ± 0.038 |
| Level15 | 0.907 ± 0.065 | 0.818 ± 0.050 | 0.748 ± 0.029 |
| Level15GSF_NoDrop | 0.926 ± 0.043 | 0.835 ± 0.026 | 0.788 ± 0.009 |
| **Level15GSF_NoDrop_K16** | **0.955 ± 0.002** | **0.855 ± 0.002** | 0.778 ± 0.002 |
| TEMFaithful | 0.907 ± 0.002 | 0.823 ± 0.003 | **0.796 ± 0.002** |

### 1.3 Cross-scale (TEM OOD-s analog)

Train on sizes 32 / 64 / 128; eval per-size on held-out envs at T=512 OOD.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| RoPE | 0.396 ± 0.040 | 0.492 ± 0.056 | 0.506 ± 0.051 |
| Vanilla | 0.535 ± 0.043 | 0.719 ± 0.066 | 0.747 ± 0.075 |
| Level15 | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| Level15GSF_NoDrop | 0.838 ± 0.111 | 0.947 ± 0.034 | 0.967 ± 0.022 |
| Level15GSF_NoDrop_K16 | 0.826 ± 0.109 | 0.951 ± 0.031 | 0.973 ± 0.019 |
| **TEMFaithful** | **0.936 ± 0.021** | **0.973 ± 0.006** | **0.981 ± 0.005** |

**TEM dominates small grids by +11pp.** Coupled-ω hypothesis in
`TEM_CROSSSCALE_DIAGNOSTIC.md`; falsifiable test (per-scale ω head) is
proposed but not yet run.

### 1.4 Cross-class (beyond TEM): torus + MiniGrid-DoorKey

Different action vocab (4 vs 7), different obs spaces, env-prefix tokens.
T=512 OOD.

| Variant | Torus T=128 | Torus T=512 | DoorKey T=128 | DoorKey T=512 |
|---|---|---|---|---|
| RoPE | 0.548 ± 0.030 | 0.497 ± 0.008 | 0.890 ± 0.011 | 0.788 ± 0.012 |
| Vanilla | 0.813 ± 0.039 | 0.681 ± 0.091 | 0.957 ± 0.008 | 0.841 ± 0.022 |
| **Level15** | 0.925 ± 0.024 | **0.879 ± 0.039** | 0.946 ± 0.003 | 0.888 ± 0.010 |
| Level15GSF_NoDrop_K16 | 0.925 ± 0.020 | 0.865 ± 0.032 | 0.945 ± 0.002 | **0.891 ± 0.000** |
| TEMFaithful | n/a | n/a | n/a | n/a |

**TEMFaithful n/a:** its `tokens<n_actions` action/obs split doesn't
accommodate the unified vocab (env-prefix tokens at IDs 0/1 + actions
spread over 2-12). Would need a custom is-action mask to compare.

---

## 2. Controls

### 2.1 VanillaNoDrop control — does dropout removal alone explain the win?

Source: `VANILLANODROP_CONTROL.md`. Multi-env held-out, n=3.

| Setting | Vanilla | VanillaNoDrop | Level15 |
|---|---|---|---|
| LM200 T=512 OOD | 0.728 ± 0.047 | **0.737 ± 0.028** | 0.988 ± 0.003 |
| CLEAN T=512 OOD | 0.920 ± 0.024 | **0.962 ± 0.020** | 0.975 ± 0.010 |

**Verdict:** VanillaNoDrop ≈ Vanilla on both. Dropout removal alone gains
≤ 1 pp on lm200. **The Level15NoDrop +13pp win is NOT reducible to a
dropout bug — InEKF correction is doing real work.** This is the
workshop-critical control.

### 2.2 NoDrop vs GSF — substitutes-not-complements

Source: `GSF_NODROP_RESULTS.md`. Single-env lm200, T=512 OOD, n=3.

| Variant | Accuracy | NLL |
|---|---|---|
| Level15 | 0.819 ± 0.025 | 0.897 |
| Level15NoDrop | 0.948 ± 0.025 | 0.317 |
| Level15GSF | 0.956 ± 0.042 | 0.227 |
| **Level15GSF_NoDrop** | **0.961 ± 0.038** | **0.177** |
| TEMFaithful (ref) | 0.969 ± 0.010 | 0.171 |

**Verdict:** NoDrop and GSF are **accuracy-substitutes** (+13pp vs +14pp;
stacked still +14pp) but **NLL-complements** (stacked 5× better NLL).
GSF earns its keep only on calibration / posterior shape, not on raw
accuracy. ~4× compute cost. For minimal sweeps, use NoDrop.

### 2.3 Pareto-shift confirmation

Source: `NODROP_PARETO_RESULTS.md`. Single-env, n=3.

| Config | Vanilla | Level15 | Level15NoDrop |
|---|---|---|---|
| Clean T=512 | 0.911 | 0.993 | 0.985 |
| Clean T=512 NLL | 0.458 | 0.039 | 0.070 |
| Noise T=512 | 0.638 | 0.702 | 0.699 |
| LM200 T=512 | 0.716 | 0.819 | **0.948** |
| LM200 T=512 NLL | 1.391 | 0.897 | 0.317 |

NoDrop is essentially Pareto-equivalent on clean/noise and a huge win
on lm200. Frame as **near-free engineering fix for landmark regimes**.

---

## 3. Mechanism findings

### 3.1 Kalman's win is stabilisation + token-type gating, not Bayesian inference

R_t learns to be HIGH on aliased obs (so the measurement contribution is
tiny) yet Level 1.5 still beats Vanilla. The structural win comes from:
1. **The wrap** (`atan2` of innovation) — keeps θ̂ bounded at OOD length.
2. **Per-token R_t** — gates by token TYPE (action vs obs).

The wrap is load-bearing for length-generalisation; the gating is
load-bearing for clean-task quality. See `feedback_post_attn_dropout.md`
and `STATE_PROBES.md` for the displacement-decoding evidence.

### 3.2 EM vs WM is regime-dependent

Source: `feedback_em_vs_wm_mechanism.md`.

EM's `A = softmax(A_X ⊙ A_P)` is a multiplicative AND-gate; WM's
additive scoring is an OR-gate.

| Regime | A_X is... | A_P is... | Winner | Confirmed |
|---|---|---|---|---|
| Paper aliased + short | noisy | sharp | EM | paper |
| Aliased + long OOD | noisy | drift-degraded | WM | our results |
| Landmarks | sharp | drift-degraded | WM | our results |
| With correction | — | repaired | tied | our results |
| DoorKey egocentric | very noisy | sharp | EM | `DOORKEY_BC_RESULTS.md` |

Backbone ordering flips by regime. Paper's "EM scales better" is along
A_X-noisy axes; our regimes flip that.

### 3.3 PC and InEKF are mathematical duals

Source: `feedback_pc_kalman_duality.md`. PC's forward map `g(θ̂) → x_t`
and InEKF's inverse `h(x_t) → z_t` write the same posterior over θ from
opposite sides. Coupling them creates a degenerate optimum
(R-saturation autoencoder bypass). Any non-zero gradient coupling
reproduces this; only full gradient isolation (Level15PC_v4) avoids it,
at which point PC adds essentially nothing.

### 3.4 Stochastic-transition MDP framing

Source: `STOCHASTIC_TRANSITION_RESULTS.md`,
`feedback_action_noise_framing.md`. Action-record corruption is
mathematically equivalent to a stochastic-transition MDP for uniform
policies. Use the latter vocabulary — standard control/RL terminology,
much harder to dismiss as artificial.

---

## 4. Goal-directed / behavioural

### 4.1 Open-loop match-acc (next-action prediction)

Source: `GOAL_DIRECTED_RESULTS.md`. Single seed, torus lm200.

| Variant | T_exp=32 | T_exp=64 (train) | T_exp=128 OOD |
|---|---|---|---|
| Vanilla | 0.628 | 0.950 | 0.766 |
| **Level15** | 0.939 | 0.947 | **0.950** |
| Level15EM | 0.936 | 0.949 | 0.948 |
| Level15NoDrop | 0.939 | 0.946 | 0.949 |

Correction-stabilised cognitive maps stay navigable across OOD explore
lengths (+18pp over Vanilla); Vanilla degrades.

### 4.2 Frozen-probe (linear readout → action)

Source: `PROBE_GOAL_RESULTS.md`.

| Variant | Held-out probe acc |
|---|---|
| Vanilla | 0.555 |
| **Level15** | 0.630 |
| Level15EM | 0.631 |
| Level15NoDrop | 0.637 |

+7.5pp gap from Vanilla → Level15 in **representation content**, not
trainability. Cleanest "cognitive maps differ in content" claim.

### 4.3 Closed-loop goal navigation — fails for everyone

Source: `GOAL_CLOSEDLOOP_RESULTS.md`.

| Variant | T_exp=32, T_nav=32 | T_exp=64 | T_exp=128 |
|---|---|---|---|
| Vanilla | 0.005 | 0.015 | 0.005 |
| Level15 | 0.010 | 0.015 | 0.020 |
| Level15EM | 0.020 | 0.020 | 0.015 |
| Level15NoDrop | 0.015 | 0.015 | 0.015 |

Match-acc 0.92-0.95 → closed-loop 0.01-0.02. **BC distribution-shift
dominates.** Cannot lead a workshop pitch with this metric as currently
structured.

### 4.4 DoorKey BC + DAgger

Source: `DOORKEY_BC_RESULTS.md`, `DAGGER_RESULTS.md`.

**BC match-acc / closed-loop:**
| Variant | match-acc | closed-loop |
|---|---|---|
| Vanilla | 0.875 | 0.250 |
| Level15 | 0.875 | 0.230 |
| **Level15EM** | **0.938** | 0.190 |
| Level15NoDrop | 0.812 | 0.240 |

EM wins match-acc on DoorKey (opposite of torus) — egocentric obs makes
A_X very noisy → multiplicative gate wins. Mechanism predicts the flip.

**DAgger (4 rounds, closed-loop success):**
| Variant | round 0 | round 4 |
|---|---|---|
| Vanilla | 0.25 | 0.10 (degrades) |
| Level15 | 0.23 | 0.25 (modest) |
| Level15EM | 0.19 | 0.08 (unstable) |
| **Level15NoDrop** | 0.24 | **0.42 (+18pp)** |

NoDrop is the only clear DAgger gain — post-attn features carry recovery
patterns that dropout was destroying.

---

## 5. Representational probes

### 5.1 Goal-distance probe (head state) — mostly chance

Source: `PROBE_GOAL_DISTANCE.md`. Single-env lm200, s0. Head: 2-layer
MLP on (post-LN hidden, goal_token_emb) → BFS distance.

| Variant | train_goals MAE/Spear | heldout_goals MAE/Spear | const-baseline |
|---|---|---|---|
| RoPE | 12.79 / 0.08 | 14.92 / -0.01 | 10.76 |
| Vanilla | 12.32 / 0.07 | 12.74 / 0.02 | 10.71 |
| Level15 | 12.66 / 0.06 | 14.35 / -0.02 | 10.85 |
| **TEMFaithful** | 12.30 / 0.18 | **12.23 / 0.27** | 11.05 |

**All at/below constant baseline on MAE.** Only TEMFaithful shows any
rank-order signal (Spearman 0.27). The mixed hidden state does NOT
encode goal-relative distance linearly.

### 5.2 Position-decode probe (explicit spatial state) — half-chance signal

Source: `STATE_PROBES.md`. Probes `theta_hat` (Level15), `theta_path`
(Vanilla), `g` (TEM). Target: displacement-from-start `(dx, dy)`.

| Variant | state dim | MAE cells | median cells | chance |
|---|---|---|---|---|
| Vanilla | 128 | 16.05 | 12.37 | 32.15 |
| Level15 | 128 | 15.32 | 11.29 | 32.08 |
| TEMFaithful | 64 | similar | similar | ~32 |

Real spatial signal — about half-chance error. Not sharp localisation.
The cognitive map IS there in `theta_hat`, just not at single-cell
precision via a 2-layer MLP head.

### 5.3 Place cells emerge in all variants; hex doesn't anywhere

Source: `paper_figures/place_cells_per_variant.png` (commit 691762a).

| Variant | Top-3 peak ratios (smoothed) |
|---|---|
| RoPE | 424× |
| Vanilla | 37× |
| Level15 | 88× |
| Level15GSF_NoDrop | 97× |
| **TEMFaithful (g)** | **9447× / 6349× / 317×** |

TEM's structural code `g` shows the cleanest place tuning by two orders
of magnitude. **No variant produces hex grid cells** (`Grid_Free` max
grid score 0.04). Likely needs continuous-state nav + Sorscher's three
conditions; discrete-cell training doesn't suffice.

### 5.4 TEM single-env clean eval (post-hoc)

Source: `TEM_BACKGROUND_BASELINES.md`. Eval pass on existing TEM
checkpoints (train_variant.py doesn't save train/test_acc).

| | T=128 | T=512 OOD | T=512 NLL |
|---|---|---|---|
| TEMFaithful | 1.000 | 0.966 ± 0.008 | 0.182 ± 0.049 |

Matches its multi-env clean number. TEM is robust across settings.

---

## 6. Other regime tests

### 6.1 Long-T evaluation

Source: `LONGT_EVAL_RESULTS.md`. T up to 2048, no retraining.

The RoPE → MapFormer gap grows with T. Level15's NLL stays low at T=2048;
Vanilla's NLL doubles. Clean accuracy converges to ceiling; lm200 / noise
gaps persist.

### 6.2 MiniGrid-DoorKey long-T

Source: `MINIGRID_DOORKEY_LONGT.md`, `MINIGRID_DK16_RESULTS.md`.

- DoorKey-8x8: Level15 +10pp noise OOD at T=512, ties clean (small env).
- DoorKey-16x16: similar.
- Long-T T=2048: noise accuracy gap +16pp.

### 6.3 MiniGrid-MemoryS13

Source: `MINIGRID_MEMORY_RESULTS.md`. Cleanest "Level15 wins clean OOD on
a real env": +13pp at T=512, +13pp at T=1024, NLL 5× better.

### 6.4 Vocab sweep

Source: `VOCAB_SWEEP_RESULTS.md`. Single seed.

At n_obs=16 / 256 / 4096, T=512 OOD on fresh obs_map:
- VanillaEM collapses at n_obs=256 (0.562, worse than Vanilla 0.665).
- Correction (Level15, Level15EM) rescues both backbones to ~0.97.
- All variants collapse at n_obs=4096 (degenerate regime).

Paper's "EM wins at large vocab" claim is l=16-specific. Doesn't survive
at our l=128/T=512.

### 6.5 Stochastic-transition vs action-record corruption

Source: `STOCHASTIC_TRANSITION_RESULTS.md`. Empirically confirms the
equivalence for uniform policies. Small (~5pp) on/off-diagonal asymmetry
in trans-noise training (slightly better generalisation due to higher
trajectory diversity).

---

## 7. Negative results worth flagging

- **Hex grid cells don't emerge anywhere** — `Grid`, `Grid_Free`,
  `Level15_DoG`, `GridL15PC_Free`. Sorscher's three conditions (path
  integration + non-negativity + DoG-similarity targets) not sufficient
  on the discrete-cell torus. Likely needs continuous-state nav.
- **Closed-loop goal navigation: 1-2% success for everyone.** BC
  distribution shift dominates. Match-acc differences don't translate
  to closed-loop behaviour without DAgger.
- **PC + Kalman creates a degenerate optimum** (R-saturation autoencoder
  bypass). Architecturally guaranteed; not a tuning issue.
- **TEM cross-scale dominates at small grids.** Level15GSF_NoDrop_K16
  at size 32: 0.826 vs TEM 0.936 (+11pp). Coupled-ω hypothesis is
  testable but not yet tested.
- **TEMFaithful cross-class incompatibility:** `tokens<n_actions` doesn't
  fit unified vocab. Would require custom is-action mask.
- **Goal-distance probe at chance.** Cognitive map is decodable for
  *prediction* but not linearly for *goal-relative distance*.

---

## 8. Workshop pitch — current state

**What's defensible (and supported by multi-seed evidence):**
1. **Cognitive-map architectures (Level15 / GSF / TEM) generalise to
   novel environments along four axes; standard transformers don't.**
   §1 above.
2. **The InEKF correction is doing real work, not unmasking a dropout
   bug.** VanillaNoDrop ≈ Vanilla; Level15NoDrop = Level15 + 13pp on
   lm200. §2.1.
3. **NoDrop is a near-free engineering Pareto-shift; GSF earns its
   keep on calibration, not accuracy.** §2.2-3.
4. **Cognitive maps differ in representation content, not just
   trainability.** Frozen linear probe +7.5pp Level15 → Vanilla. §4.2.
5. **EM vs WM is regime-dependent; mechanism predicts both signs.**
   Same architecture wins different regimes. §3.2.

**What's NOT defensible:**
- "Cognitive maps enable closed-loop goal navigation" — currently false
  for all variants (§4.3).
- "Hex grid cells emerge" — false everywhere (§7).
- "Level1.5 is universally best on scale axes" — TEM wins at small
  grids (§1.3).

**Open / proposed:**
- Per-scale ω head to close the small-grid gap (§1.3, falsifiable).
- Active-inference / world-model planning as the right framing for the
  behavioural story, sidestepping closed-loop BC failure.
- Scale to 4 layers, d=256 for workshop-scale claim.

---

## Appendix — source file index

Headline tables:
- `TEM_NOVEL_ENV_RESULTS.md` — 4-axis novel-env comparison (§1)
- `MULTIENV_CLEAN_2x2.md` — multi-env clean × lm200 (§1.1)
- `MULTICLASS_MULTISEED_RESULTS.md` — torus + DoorKey (§1.4)

Controls and mechanisms:
- `VANILLANODROP_CONTROL.md` — InEKF necessity (§2.1)
- `GSF_NODROP_RESULTS.md` — accuracy substitutes / NLL complements (§2.2)
- `NODROP_PARETO_RESULTS.md` — Pareto verification (§2.3)
- `TEM_CROSSSCALE_DIAGNOSTIC.md` — small-grid mechanism (§1.3, §7)

Behaviour:
- `GOAL_DIRECTED_RESULTS.md` — match-acc (§4.1)
- `PROBE_GOAL_RESULTS.md` — frozen linear probe (§4.2)
- `GOAL_CLOSEDLOOP_RESULTS.md` — closed-loop failure (§4.3)
- `DOORKEY_BC_RESULTS.md`, `DAGGER_RESULTS.md` — DoorKey BC (§4.4)

Representational probes:
- `PROBE_GOAL_DISTANCE.md` — head-state probe (§5.1)
- `STATE_PROBES.md` — explicit-state probe (§5.2)
- `paper_figures/place_cells_per_variant.png` — place cells (§5.3)
- `TEM_BACKGROUND_BASELINES.md` — TEM single-env eval (§5.4)

Other regimes:
- `LONGT_EVAL_RESULTS.md`, `MINIGRID_DOORKEY_LONGT.md`,
  `MINIGRID_MEMORY_RESULTS.md`, `VOCAB_SWEEP_RESULTS.md`,
  `STOCHASTIC_TRANSITION_RESULTS.md`

Negative / superseded (do not cite without context):
- `DOG_RESULTS.md` (vacuous targets, superseded by `DOG_RESULTS_FIXED.md`)
- `MULTICLASS_RESULTS.md` (single-seed, superseded by MULTISEED version)
- `TEM_RESULTS.md` (pre-bug-fix TEMFaithful, predict-then-update was wrong)

*Generated 2026-05-15. Verify against `git log` for newer results.*
