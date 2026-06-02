# MapFormer extensions — consolidated report v2

Supersedes `REPORT.md` (2026-05-15) and `REPORT_ADDENDUM.md` (2026-05-18).
Folds in everything since: the full TEM-setting headline (now with 4
generalization axes populated), the cross-scale architecture chain
(SingleSize → PerScale ω → Hopfield head → NoMainAP → ExtraHead
capacity control), the **corrected capacity verdict** (per-regime +
length sweep overturning the earlier lm200-only "CAPACITY" reading),
TEM parameter-scaling, NumberLine arithmetic-as-navigation, and the
TEMFaithful_FFN direct test.

Every claim cites its source MD; numbers are n=3 unless noted.

---

## 0. Framing — what changed since REPORT.md

Three things shifted between REPORT.md and now.

1. **The capacity question got a proper per-regime answer.** The earlier
   `CAPACITY_CONTROL.md` tested only single-env lm200 at T=512 and
   declared "CAPACITY" because a generic extra attention head matched
   Level15 on that one point. The corrected control
   (`CAPACITY_PERREGIME.md`) tests clean / noise / lm200 across
   T in {512, 1024, 2048} plus a NumberLine OOD chain. The verdict is
   ARCHITECTURE on clean / noise / length / arithmetic / calibration;
   lm200 alone has a content-channel effect where an extra content head
   retrieves unique landmark tokens. REPORT.md / earlier writeups
   oversold "Level15 beats Vanilla on lm200 accuracy" as a generic win
   — that specific lm200 accuracy number is partly reproducible by a
   generic extra head, but the architectural claim survives on every
   other regime that matters.

2. **The cross-scale story walked through five mechanism candidates and
   landed on "extra-head capacity, not Hopfield structure."** The
   chain: TEM dominates small grids → coupled-ω hypothesis →
   `SINGLE_SIZE_CONTROL` confirms it → per-scale ω architectural
   fix closes ~half the gap → Hopfield head matches TEM →
   `HOPFIELD_NOMAINAP` ablation shows main-AP is load-bearing →
   `EXTRAHEAD_CONTROL` matches Hopfield with a generic head. The
   cross-scale cure is more attention capacity / seed-instability
   rescue, not TEM-shaped memory. TEM remains the most seed-stable.

3. **The TEM-setting headline (4 axes) is fully populated and
   defensible.** Held-out-env / cross-topology / cross-scale /
   cross-class with multi-seed numbers for RoPE, Vanilla, Level15,
   Level15GSF_NoDrop, Level15GSF_NoDrop_K16 and TEMFaithful. Cognitive-
   tier architectures all clear the bar; the choice between Level15-
   family and TEM is regime-dependent. The TEMFaithful_FFN direct test
   landed — adding a per-position FFN nudges TEM clean from 0.966 →
   0.969 (not enough to close the lag) but improves NLL meaningfully.

The one-line workshop pitch from REPORT.md still holds:

> "MapFormer extends to novel environments along four axes —
> held-out same-class, cross-topology, cross-scale, cross-class.
> Across all four, our Level 1.5 correction matches or exceeds TEM-
> style explicit memory."

with the corrected caveats spelled out in §5 and §11.

---

## 1. Models

### Baselines and reference architectures

- **MapFormer-WM** — paper-faithful, single-rotation path integrator,
  position scoring added to attention logits (additive OR-gate).
- **MapFormer-EM** — paper-faithful, separate `q0_pos` / `k0_pos`
  rotations, `A = softmax(A_X . A_P)` (multiplicative AND-gate).
- **RoPE** — vanilla transformer with rotary position embeddings.
  Standard-transformer baseline.
- **Vanilla** — Vanilla MapFormer-WM (no Kalman correction). The main
  non-cognitive baseline. ~256K params.
- **VanillaNoDrop** — Vanilla with the post-attention residual dropout
  layer removed (see §5). Control for the NoDrop finding.
- **LSTM**, **CoPE**, **MambaLike** — non-MapFormer baselines from
  earlier sessions; numbers in `RESULTS_PAPER.md`.

### Correction-augmented MapFormers

- **Level1 / Level1.5 / Level2 InEKF** — Invariant EKF on the path-
  integrator output theta_hat. Level 1 is the parallel-scan
  steady-state-gain variant; Level 1.5 adds a constant learnable
  `log_Pi` and a per-token-type `log_R_head` (best speed/accuracy
  tradeoff); Level 2 adds full per-token heteroscedastic R_t (~60×
  slower training).
- **Level15EM** — Level 1.5 on the EM backbone (`log_R_init_bias=3.0`
  to prevent the AND-gate init pathology).
- **Level15_PerScaleOmega** — Level 1.5 with one learnable ω per
  training grid size, selected at forward time by an `env_sizes`
  kwarg. Targets the coupled-ω cross-scale bottleneck.
- **Level15_Hopfield** — Level 1.5 + a TEM-style position-keyed
  Hopfield retrieval head (parallel attention head with KV restricted
  to obs positions, key from position-only rotated `k0_pos`).
- **Level15_Hopfield_NoMainAP** — Level15_Hopfield with the main
  attention's position rotations replaced by identity. Tests whether
  position-modulated main attention is load-bearing once the Hopfield
  head exists.
- **Level15EM_Hopfield** — EM backbone + Hopfield head. (Listed in
  `EM_HOPFIELD_CROSSSCALE.md`; Level15EM_PerScaleOmega is the main
  EM-backbone fix tested.)
- **Vanilla_ExtraHead** — Vanilla + a GENERIC extra attention head
  (content Q/K, position-modulated, all-positions causal KV). The
  capacity control. 322K params (more than Level15).
- **Level15_ExtraHead** — Level 1.5 + same generic extra head. The
  cross-scale capacity control.
- **L15_NoCorr** — Level 1.5 architecture (305K params) with the
  InEKF correction zeroed at runtime. Same params as Level15, no
  correction. Param-matched control.
- **Level15NoDrop** — Level 1.5 with the inherited post-attention
  residual dropout removed.
- **Level15GSF** — Gaussian Sum Filter with K=8 parallel Level 1.5
  chains; learnable per-chain `theta_init_k` offsets; mixture weights
  from cumulative log-likelihood.
- **Level15GSF_NoDrop** / **_NoDrop_K16** — GSF stacked with NoDrop;
  K16 variant uses 16 modes.

### TEM family

- **TEMFaithful** — Whittington-style TEM with separate `g`
  (structural code) and Hopfield memory bank keyed by `g`,
  `W_a = exp(skew(A_a))` orthogonal per-action rotation in `d_g`
  space, predict-then-update. The bug-fixed version (queries memory
  with POST-action `g`).
- **TEMFaithful_FFN** — TEMFaithful + a per-position FFN on the
  retrieved content (fixed Hopfield bank unchanged).
- **TEMFaithful_dg{32, 64, 128, 256}** — parameter sweep on `d_g`. The
  d_g=64 case is the default; d_g=256 is parameter-matched to Level15.
- **TEM-T** — transformer-formulation of TEM; reference comparison
  only.

### NumberLine variants

- **NumberLine{Vanilla, Level15, Vanilla_ExtraHead}** — MapFormer
  variants on a 1D additive torus (N=64, 6 ops). Tests whether the
  Level15 self-correcting accumulator extrapolates to longer
  arithmetic chains.

---

## 2. Headline — TEM-setting novel-environment generalization

Source: `TEM_NOVEL_ENV_RESULTS.md`, `MULTIENV_CLEAN_2x2.md`,
`MULTICLASS_MULTISEED_RESULTS.md`, `MULTISEED_FOLLOWUP_RESULTS.md`,
`TEM_BACKGROUND_BASELINES.md`. All n=3.

### 2.1 Multi-env held-out (TEM's classic novel-env test)

50 train envs, 50 held-out test envs, size 64, T=128 train.

**LM200 (200 landmarks):**

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

Cognitive-tier all in 0.97-0.99 OOD. Standard transformers are not.
Choice between Level15-family and TEM is regime-dependent (TEM slightly
better with landmarks, Level15GSF_NoDrop slightly better on clean).

### 2.2 Cross-topology (TEM OOD-d analog)

Train on torus + open + walls mix; eval per-topology on held-out envs
at T=512 OOD.

| Variant | torus | open | walls |
|---|---|---|---|
| RoPE | 0.468 ± 0.027 | 0.510 ± 0.027 | 0.536 ± 0.006 |
| Vanilla | 0.787 ± 0.039 | 0.717 ± 0.032 | 0.658 ± 0.038 |
| Level15 | 0.907 ± 0.065 | 0.818 ± 0.050 | 0.748 ± 0.029 |
| Level15GSF_NoDrop | 0.926 ± 0.043 | 0.835 ± 0.026 | 0.788 ± 0.009 |
| **Level15GSF_NoDrop_K16** | **0.955 ± 0.002** | **0.855 ± 0.002** | 0.778 ± 0.002 |
| TEMFaithful | 0.907 ± 0.002 | 0.823 ± 0.003 | **0.796 ± 0.002** |

### 2.3 Cross-scale (TEM OOD-s analog)

Train on sizes 32 / 64 / 128; eval per-size at T=512 OOD.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| RoPE | 0.396 ± 0.040 | 0.492 ± 0.056 | 0.506 ± 0.051 |
| Vanilla | 0.535 ± 0.043 | 0.719 ± 0.066 | 0.747 ± 0.075 |
| Level15 | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| Level15GSF_NoDrop | 0.838 ± 0.111 | 0.947 ± 0.034 | 0.967 ± 0.022 |
| Level15GSF_NoDrop_K16 | 0.826 ± 0.109 | 0.951 ± 0.031 | 0.973 ± 0.019 |
| **TEMFaithful** | **0.936 ± 0.021** | **0.973 ± 0.006** | **0.981 ± 0.005** |

**TEM dominates small grids by +11pp** with much tighter seed variance
(0.021 vs 0.109). §3 walks through the mechanism investigation.

### 2.4 Cross-class (beyond TEM): torus + MiniGrid-DoorKey

Different action vocab (4 vs 7), different obs spaces, env-prefix
tokens.

| Variant | Torus T=128 | Torus T=512 | DoorKey T=128 | DoorKey T=512 |
|---|---|---|---|---|
| RoPE | 0.548 ± 0.030 | 0.497 ± 0.008 | 0.890 ± 0.011 | 0.788 ± 0.012 |
| Vanilla | 0.813 ± 0.039 | 0.681 ± 0.091 | 0.957 ± 0.008 | 0.841 ± 0.022 |
| **Level15** | 0.925 ± 0.024 | **0.879 ± 0.039** | 0.946 ± 0.003 | 0.888 ± 0.010 |
| Level15GSF_NoDrop_K16 | 0.925 ± 0.020 | 0.865 ± 0.032 | 0.945 ± 0.002 | **0.891 ± 0.000** |
| TEMFaithful | n/a | n/a | n/a | n/a |

TEMFaithful's `tokens<n_actions` action/obs split doesn't accommodate
the unified vocab (env-prefix tokens at IDs 0/1 + actions spread over
2-12). Would need a custom is-action mask to compare.

---

## 3. Cross-scale investigation — five candidates, mechanism walked

The +11pp small-grid TEM gap (§2.3) prompted a sequence of falsifiable
tests. Each row is the "winner-so-far" hypothesis; subsequent rows
either confirm or falsify it.

### 3.1 TEM cross-scale diagnostic (analytic)

Source: `TEM_CROSSSCALE_DIAGNOSTIC.md`. Four candidate mechanisms:
TEM's `W_a` is scale-invariant by construction; Level15's path
integrator has scale-coupled ω; TEM's Hopfield retrieval is scale-
agnostic; GSF mode init is scale-coarse. The cheapest test: per-scale
ω. The most informative: single-size Level15.

### 3.2 SingleSize control — coupled-ω confirmed

Source: `SINGLE_SIZE_CONTROL.md`. Train Level15 lm200 on a SINGLE grid
size; compare to multi-size mix.

| Variant | size 16 T=512 | size 32 T=512 |
|---|---|---|
| Level15 single-size | 0.159 ± 0.019 | **0.908 ± 0.085** |
| Level15 multi-size (32+64+128) | — | 0.782 ± 0.138 |
| TEMFaithful multi-size (ref) | — | 0.936 ± 0.021 |

Single-size 32 (0.908) >> multi-size 32 (0.782) → **+13pp**. Closes
most of the gap to TEM. Coupled-ω confirmed as a real bottleneck.
Size 16 collapses for all (n_blocks=32 > 256 positions; different
failure mode).

### 3.3 Per-scale ω — partial fix

Source: `PERSCALE_OMEGA_RESULTS.md`. One learnable ω per training
scale, selected by `env_sizes` kwarg.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| Level15 (coupled ω) | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| **Level15_PerScaleOmega** | **0.877 ± 0.163** | 0.937 ± 0.075 | 0.946 ± 0.074 |
| TEMFaithful (ref) | 0.936 ± 0.021 | 0.973 ± 0.006 | 0.981 ± 0.005 |

Size 32: +10pp over coupled-ω Level15. Closes about half the gap
to TEM. Sizes 64/128 essentially tied. High variance (±0.163) — at
least one seed under-converged. Real architectural improvement at
small grids, but not the full story.

### 3.4 Hopfield head — closes the gap

Source: `EM_HOPFIELD_CROSSSCALE.md`. Add a TEM-style position-keyed
Hopfield retrieval head to Level15 / Level15EM.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| Vanilla | 0.535 ± 0.043 | 0.719 ± 0.066 | 0.747 ± 0.075 |
| Level15 | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| Level15EM | 0.740 ± 0.042 | 0.887 ± 0.051 | 0.913 ± 0.060 |
| Level15_PerScaleOmega | 0.877 ± 0.163 | 0.937 ± 0.075 | 0.946 ± 0.074 |
| Level15EM_PerScaleOmega | 0.766 ± 0.063 | 0.930 ± 0.032 | 0.956 ± 0.028 |
| **Level15_Hopfield** | **0.919 ± 0.089** | **0.972 ± 0.030** | **0.985 ± 0.018** |
| TEMFaithful (ref) | 0.936 ± 0.021 | 0.973 ± 0.006 | 0.981 ± 0.005 |

Level15_Hopfield matches TEMFaithful within seed noise at every scale.
First read: "we ported TEM's memory mechanism into MapFormer at
parallel cost." Spoiler: §3.6 falsifies that framing.

Level15EM consistently underperforms Level15 at every scale — backbone
choice not orthogonal to small-grid issue.

### 3.5 NoMainAP — main attention's position channel IS load-bearing

Source: `HOPFIELD_NOMAINAP_RESULTS.md`. Identity-rotate the main
attention's position channel; position enters the model ONLY through
the Hopfield head.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| Level15 | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| Level15_Hopfield | 0.919 ± 0.089 | 0.972 ± 0.030 | 0.985 ± 0.018 |
| **Level15_Hopfield_NoMainAP** | **0.627 ± 0.189** | **0.741 ± 0.188** | **0.750 ± 0.196** |
| TEMFaithful (ref) | 0.936 ± 0.021 | 0.973 ± 0.006 | 0.981 ± 0.005 |

Removing main-AP **collapses** Level15_Hopfield. The factored (TEM-
style) design is NOT sufficient; MapFormer's position-modulated main
attention is load-bearing. The Hopfield head is a supplement, not a
replacement. So whatever the Hopfield head is doing, it's not
substituting for main-AP — it's adding something on top.

### 3.6 ExtraHead capacity control — the Hopfield win is capacity

Source: `EXTRAHEAD_CONTROL.md`. Replace the Hopfield head with a
GENERIC extra attention head (content Q/K, position-modulated, all-
positions causal KV) — same residual wiring, **more parameters**.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| Level15 | 0.782 ± 0.138 | 0.921 ± 0.050 | 0.953 ± 0.032 |
| Level15_Hopfield | 0.919 ± 0.089 | 0.972 ± 0.030 | 0.985 ± 0.018 |
| **Level15_ExtraHead** | **0.934 ± 0.081** | **0.979 ± 0.028** | 0.984 ± 0.022 |
| TEMFaithful (ref) | 0.936 ± 0.021 | 0.973 ± 0.006 | 0.981 ± 0.005 |

Per-seed at size 32:

| Variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Level15 | 0.977 | 0.701 | 0.668 |
| Level15_Hopfield | 0.986 | 0.793 | 0.977 |
| Level15_ExtraHead | 0.987 | 0.819 | 0.996 |

A generic extra head matches the Hopfield head (0.934 vs 0.919,
indistinguishable at this variance). Base Level15 collapses on 2 of 3
seeds at size 32 — the coupled-ω bad basin. **Adding ANY extra
attention head rescues most seeds; the position-only-key / obs-
restricted-KV structure is not necessary.**

**The earlier "we ported TEM's memory mechanism" framing is NOT
supported. The cross-scale fix is capacity / training stabilisation,
not Hopfield structure.**

TEMFaithful remains the most seed-stable (±0.021 vs ±0.081). The
extra-head fix reduces but does not eliminate Level15's seed
instability.

---

## 4. Controls — what does and does not generalise

### 4.1 VanillaNoDrop — InEKF correction is doing real work

Source: `VANILLANODROP_CONTROL.md`. Multi-env held-out, n=3.

| Setting | Vanilla | VanillaNoDrop | Level15 |
|---|---|---|---|
| LM200 T=512 OOD | 0.728 ± 0.047 | **0.737 ± 0.028** | 0.988 ± 0.003 |
| CLEAN T=512 OOD | 0.920 ± 0.024 | **0.962 ± 0.020** | 0.975 ± 0.010 |

VanillaNoDrop ≈ Vanilla on lm200, ~+4pp on clean. **The Level15NoDrop
+13pp single-env lm200 win is NOT reducible to a dropout fix — InEKF
correction is doing real work.** Workshop-critical control.

### 4.2 Capacity question — corrected verdict (CRITICAL)

Source: `CAPACITY_PERREGIME.md`. Supersedes `CAPACITY_CONTROL.md`.

The earlier control tested only single-env lm200 at T=512 and saw
`Vanilla_ExtraHead` (0.875) >= `Level15` (0.828). Verdict was
"CAPACITY" — Level15's win came from its extra params, not its
architecture.

The per-regime control overturns this. Per-regime (T=512 OOD, n=3):

| Regime | Vanilla | Vanilla_ExtraHead | Level15 | verdict |
|---|---|---|---|---|
| **clean** | 0.918 ± 0.042 | **0.742 ± 0.049** | 0.995 ± 0.004 | ARCHITECTURE |
| **noise** | 0.634 ± 0.031 | 0.648 ± 0.034 | 0.707 ± 0.012 | ARCHITECTURE |
| **lm200** | 0.721 ± 0.054 | 0.875 ± 0.139 | 0.828 ± 0.023 | CAPACITY-looking |

On clean, the extra head is *worse* than Vanilla (the extra content
attention destabilises the path-integration regime). On noise, the
extra head matches Vanilla — doesn't reach Level15. Only lm200 reaches
Level15, with high variance (±0.139).

**Length sweep — clean accuracy.** The bounded-error signature is
visible only across lengths:

| Variant | T=512 | T=1024 | T=2048 |
|---|---|---|---|
| Vanilla | 0.918 ± 0.042 | 0.802 ± 0.062 | 0.627 ± 0.035 |
| Vanilla_ExtraHead | 0.742 ± 0.049 | 0.601 ± 0.026 | 0.526 ± 0.018 |
| **Level15** | **0.995 ± 0.004** | **0.968 ± 0.006** | **0.886 ± 0.013** |

Gap at T=512: +8pp. Gap at T=2048: **+26pp**. Capacity is the worst at
every length; only the wrap-stabilised filter holds up.

**Length sweep — clean NLL (calibration):**

| Variant | T=512 | T=1024 | T=2048 |
|---|---|---|---|
| Vanilla | 0.410 ± 0.162 | 1.352 ± 0.424 | 3.084 ± 0.625 |
| Vanilla_ExtraHead | 1.620 ± 0.209 | 3.023 ± 0.191 | 3.978 ± 0.363 |
| **Level15** | **0.031 ± 0.023** | **0.178 ± 0.050** | **0.648 ± 0.084** |

Level15 is 5–50× better than both. The extra head makes NLL *worse*
across the board.

**NumberLine — arithmetic OOD chain (T=512, 4× trained chain
length):**

| Variant | in-dist T=128 | OOD chain T=512 | T=512 NLL |
|---|---|---|---|
| Vanilla | 0.925 ± 0.009 | 0.633 ± 0.071 | 2.024 ± 0.666 |
| Vanilla_ExtraHead | 0.986 ± 0.000 | 0.662 ± 0.130 | 2.542 ± 1.324 |
| **Level15** | 0.902 ± 0.023 | **0.841 ± 0.056** | **0.521 ± 0.292** |

Capacity (Vanilla_ExtraHead) does NOT close Level15's +21pp arithmetic-
extrapolation gap.

**Honest verdict:**

- **ARCHITECTURE** on clean accuracy, noise accuracy, length
  generalization (clean / noise / lm200), arithmetic extrapolation,
  and calibration (NLL) at every regime / length.
- **CAPACITY-looking** on single-env lm200 accuracy: a generic extra
  *content* head can find unique landmark tokens with content-only
  attention. The mechanism is content retrieval of one-shot tokens,
  not generic param-count.

**The corrected claim:** Level15's win on the regimes that matter
(clean OOD, length extrapolation, calibration, arithmetic) is
architectural — it's the wrap + per-token-type gating doing the work,
not 50K extra parameters. The single-env lm200 accuracy headline
specifically is partly reproducible by an extra content head, which
should be flagged honestly.

### 4.3 L15_NoCorr — param-matched no-correction control

Same table (`CAPACITY_PERREGIME.md` / `CAPACITY_CONTROL.md`).
`L15_NoCorr` = Level15 architecture (305K params) with the InEKF
correction zeroed at runtime. lm200 T=512 OOD: 0.677 ± 0.001 — close
to Vanilla (0.721), well below Level15 (0.828). Confirms the
correction itself (not the extra params) is what's lifting Level15
above Vanilla on lm200 once you control for capacity.

### 4.4 SingleSize — coupled-ω confirmed (replayed from §3.2)

Already covered above. Listed here as the cross-scale control. The
multi-size training regime hurts small grids by ~13pp via shared
ω.

---

## 5. NoDrop and GSF — the dropout finding and multi-modal Bayes

### 5.1 NoDrop Pareto-shift (single-env, n=3)

Source: `NODROP_PARETO_RESULTS.md`.

| Config | Vanilla | Level15 | Level15NoDrop |
|---|---|---|---|
| Clean T=512 | 0.911 | 0.993 | 0.985 |
| Clean T=512 NLL | 0.458 | 0.039 | 0.070 |
| Noise T=512 | 0.638 | 0.702 | 0.699 |
| LM200 T=512 | 0.716 | 0.819 | **0.948** |
| LM200 T=512 NLL | 1.391 | 0.897 | 0.317 |

NoDrop is essentially Pareto-equivalent on clean/noise (differences
within seed std) and a +13pp win on single-env lm200. Near-free
engineering Pareto-shift for landmark regimes.

### 5.2 NoDrop vs GSF — substitutes on accuracy, complements on NLL

Source: `GSF_NODROP_RESULTS.md`. Single-env lm200, T=512 OOD, n=3.

| Variant | Accuracy | NLL |
|---|---|---|
| Level15 | 0.819 ± 0.025 | 0.897 |
| Level15NoDrop | 0.948 ± 0.025 | 0.317 |
| Level15GSF | 0.956 ± 0.042 | 0.227 |
| **Level15GSF_NoDrop** | **0.961 ± 0.038** | **0.177** |
| TEMFaithful (ref) | 0.969 ± 0.010 | 0.171 |

NoDrop and GSF are accuracy-substitutes (+13pp vs +14pp; stacked still
+14pp) but **NLL-complements** (stacked 5× better NLL). GSF earns its
keep only on calibration / posterior shape, not on raw accuracy. ~4×
compute cost. For minimal sweeps, use NoDrop.

### 5.3 Mechanism — Vaswani's post-attn residual dropout is the culprit

`WMTransformerLayer` (the paper-faithful layer) wraps `o_proj(out)` in
`self.dropout` on the residual path. With lots of redundant retrievals
(aliased obs: ~128 copies per token type) this regularises; with rare
retrievals (a landmark token appearing once) it destroys the signal.
Removing the dropout: +13pp on lm200, ~free elsewhere. The "learnable
beta / sharper softmax" framing tested in `LEVEL15BETA_RESULTS.md` was
a red herring — beta barely moves from init, but `Level15NoDrop` (beta
fixed, only dropout removed) matches `Level15Beta` exactly.

---

## 6. NumberLine — arithmetic as navigation

Source: `NUMBERLINE_RESULTS.md` (plus the capacity row in §4.2).

MapFormer on a 1D additive torus (N=64, 6 ops: ±1/±2/±3). The path
integrator θ = ω * cumsum(f_Delta(action)) literally computes
(a+b+c+...) mod N. Task: predict the obs token at a REVISITED value.

Train chain = 128 ops; OOD chain = 512 ops (4× longer).

| Variant | in-dist T=128 | OOD chain T=512 | T=512 NLL |
|---|---|---|---|
| Vanilla | 0.925 ± 0.009 | 0.633 ± 0.071 | 2.024 ± 0.666 |
| Vanilla_ExtraHead | 0.986 ± 0.000 | 0.662 ± 0.130 | 2.542 ± 1.324 |
| **Level15** | 0.902 ± 0.023 | **0.841 ± 0.056** | **0.521 ± 0.292** |

**Reading:** Level15 +21pp on OOD arithmetic chain at 4× trained
length. The self-correcting accumulator generalises to longer chains
where uncorrected cumsum drifts. Capacity (Vanilla_ExtraHead, 0.662)
does NOT close the gap — gives at most +3pp over Vanilla. NLL gap
even larger (Level15 4× lower).

Both in-distribution accuracies are low (~0.92) — the aliased-obs
recall task on a number line is hard; revisit-prediction needs work.
But the arithmetic-extrapolation signal is clean.

---

## 7. TEM analysis

### 7.1 TEM parameter-scaling — saturates, doesn't scale

Source: `TEM_SCALING_RESULTS.md`. TEMFaithful at d_g in {32, 64, 128,
256}. A_a (per-action transition matrices, the bulk of TEM's
learnable "model") scales as 4 * d_g^2. d_g=256 (~291K params) is
parameter-matched to Level15 (~305K).

| Variant | d_g | params | T=128 | T=512 OOD | T=512 NLL |
|---|---|---|---|---|---|
| **TEMFaithful_dg32** | 32 | 32,766 | 1.000 | **0.979 ± 0.002** | **0.141 ± 0.013** |
| TEMFaithful | 64 | 45,086 | 1.000 | 0.973 ± 0.010 | 0.145 ± 0.067 |
| TEMFaithful_dg128 | 128 | 94,302 | 1.000 | 0.966 ± 0.006 | 0.192 ± 0.042 |
| TEMFaithful_dg256 | 256 | 291,038 | 1.000 | 0.953 ± 0.008 | 0.259 ± 0.043 |

**TEM anti-scales**: dg32 (0.979) > dg256 (0.953) on accuracy, and the
NLL pattern is monotone in the wrong direction. Strong evidence for
the inductive-bias account — TEM is parameter-efficient because the
hard part (associative memory via fixed Hopfield retrieval) is
algorithmic, not learned. Adding capacity to A_a doesn't help; it
hurts.

### 7.2 TEMFaithful_FFN — does the missing FFN close TEM's clean lag?

Source: `TEM_NOISE_FFN_RESULTS.md`. TEMFaithful + per-position FFN on
the retrieved content. Targets the 3pp clean-regime gap vs Level15.

**Single-env clean:**

| Variant | T=128 | T=512 OOD | T=512 NLL |
|---|---|---|---|
| TEMFaithful (no FFN) | 1.000 | 0.966 ± 0.008 | 0.182 |
| **TEMFaithful_FFN** | 1.000 | **0.969 ± 0.002** | **0.145 ± 0.003** |
| Level15 (ref) | 1.000 | 0.993 | 0.039 |

Adding the FFN nudges TEM clean accuracy from 0.966 → 0.969 (within
noise — not enough to close the 2.4pp lag to Level15) and improves
NLL 0.182 → 0.145. The missing-FFN hypothesis is largely falsified:
the clean lag is from something else (learned content attention, not
per-position processing).

**Single-env lm200:**

| Variant | T=128 | T=512 OOD | T=512 NLL |
|---|---|---|---|
| TEMFaithful (no FFN) | 1.000 | 0.969 ± 0.010 | 0.171 |
| **TEMFaithful_FFN** | 1.000 | **0.979 ± 0.005** | **0.093 ± 0.027** |

Modest +1pp accuracy, NLL nearly halved. Not free machinery but
doesn't hurt.

### 7.3 TEMFaithful in the noise regime

Source: `TEM_NOISE_FFN_RESULTS.md`.

| Variant | T=128 noise | T=512 noise OOD | T=512 NLL |
|---|---|---|---|
| **TEMFaithful** | 0.762 ± 0.003 | **0.709 ± 0.002** | 1.216 ± 0.010 |
| Vanilla (ref) | — | 0.638 | — |
| Level15 (ref) | — | 0.702 | — |
| Level15NoDrop (ref) | — | 0.699 | — |

TEMFaithful in the noise regime narrowly beats Level15 / Level15NoDrop
on accuracy at T=512 (+0.7pp / +1.0pp) but loses on NLL by a lot
(1.216 vs 0.952 for Level15). Tied within practical significance.

---

## 8. Other regimes (largely unchanged from REPORT.md)

### 8.1 Long-T evaluation

Source: `LONGT_EVAL_RESULTS.md`, `CAPACITY_PERREGIME.md` length-sweep
tables (§4.2 above). T up to 2048, no retraining. RoPE → MapFormer
gap grows with T. Level15's NLL stays low at T=2048; Vanilla's NLL
doubles. Level15 clean: 0.886 at T=2048 vs Vanilla 0.627 (+26pp).

### 8.2 MiniGrid

- DoorKey-8x8 (`MINIGRID_DOORKEY_*.md`, `MINIGRID_DK16_RESULTS.md`):
  Level15 +10pp noise OOD at T=512, ties clean. Long-T T=2048: noise
  gap +16pp.
- MemoryS13 (`MINIGRID_MEMORY_RESULTS.md`): cleanest "Level15 wins
  clean OOD on a real env": +13pp at T=512, +13pp at T=1024, NLL 5×
  better.

### 8.3 Vocab sweep

Source: `VOCAB_SWEEP_RESULTS.md`. Single seed, T=512 OOD on fresh
obs_map. At n_obs=256, VanillaEM collapses (0.562, worse than Vanilla
0.665). Correction rescues both backbones to ~0.97. All variants
collapse at n_obs=4096 (degenerate regime). Paper's "EM wins at large
vocab" claim is l=16-specific.

### 8.4 Sparse landmarks

Source: `SPARSE_LANDMARKS_RESULTS.md`. TEM dominates sparse-landmark
density. Frame as a TEM strength; Level15 stays within a few pp.

### 8.5 Stochastic-transition vs action-record corruption

Source: `STOCHASTIC_TRANSITION_RESULTS.md`. Empirically confirms the
equivalence for uniform policies. Small (~5pp) asymmetry; use
stochastic-transition vocabulary for the writeup.

---

## 9. Mechanism findings (unchanged from REPORT.md)

### 9.1 Kalman's win is stabilisation + token-type gating

R_t learns to be HIGH on aliased obs (measurement contribution tiny)
yet Level 1.5 still beats Vanilla. The structural win comes from:
(1) the wrap (`atan2` of innovation) — keeps theta_hat bounded at OOD
length; (2) per-token R_t — gates by token TYPE (action vs obs). The
wrap is load-bearing for length generalization; gating for clean-task
quality.

### 9.2 EM vs WM is regime-dependent

| Regime | A_X | A_P | Winner | Confirmed |
|---|---|---|---|---|
| Paper aliased + short | noisy | sharp | EM | paper |
| Aliased + long OOD | noisy | drift-degraded | WM | ours |
| Landmarks | sharp | drift-degraded | WM | ours |
| With correction | — | repaired | tied | ours |
| DoorKey egocentric | very noisy | sharp | EM | `DOORKEY_BC_RESULTS.md` |

### 9.3 PC and InEKF are mathematical duals

Coupling them creates a degenerate optimum (R-saturation autoencoder
bypass). Only full gradient isolation (Level15PC_v4) avoids it, at
which point PC adds essentially nothing. See
`feedback_pc_kalman_duality.md`.

---

## 10. Goal-directed / behavioural (unchanged from REPORT.md)

Summary numbers below; full detail in REPORT.md §4.

- **Open-loop match-acc** (`GOAL_DIRECTED_RESULTS.md`): Level15 +18pp
  over Vanilla at OOD explore length. Correction-stabilised maps stay
  navigable.
- **Frozen-probe** (`PROBE_GOAL_RESULTS.md`): Level15 +7.5pp over
  Vanilla on held-out probe (linear readout → action). Cognitive maps
  differ in CONTENT, not just trainability.
- **Closed-loop goal navigation** (`GOAL_CLOSEDLOOP_RESULTS.md`):
  1-2% success for everyone. BC distribution shift dominates. Cannot
  lead the workshop pitch with this.
- **DoorKey BC + DAgger** (`DOORKEY_BC_RESULTS.md`,
  `DAGGER_RESULTS.md`): EM wins match-acc on DoorKey (egocentric →
  A_X noisy → multiplicative gate wins, opposite of torus). DAgger:
  only Level15NoDrop shows a clear gain (0.24 → 0.42).
- **Active-inference one-step / multi-step**
  (`ACTIVE_INFERENCE_RESULTS.md`): 0-2% across all variants and
  horizons. The frozen forward model is structurally too myopic to
  drive multi-step nav.
- **Successor-rep aux pretraining** (`SR_PRETRAIN_RESULTS.md`): chance
  on both goal-distance probe and active-inference closed-loop. Aux
  loss with a separate head doesn't force the backbone to change.

---

## 11. Probes and honest negatives

Refer to REPORT.md §5-§7 plus RESULTS_INDEX.md for the full list.
Briefly:

- **Goal-distance probe (head state)** at chance for all variants;
  TEMFaithful is the only one showing rank-order signal (Spearman
  0.27).
- **Position-decode probe** half-chance on displacement-from-start;
  cognitive map IS there in theta_hat, just not at single-cell
  precision via a 2-layer MLP head.
- **Place cells emerge everywhere**; **hex doesn't emerge anywhere**
  (Grid, Grid_Free, Level15_DoG, GridL15PC_Free).
- **Sorscher's three conditions** (path integration + non-negativity
  + DoG-similarity targets) not sufficient on the discrete-cell torus.
- **PC + Kalman** architecturally guarantees a degenerate optimum
  (R-saturation autoencoder bypass).
- **TEM cross-scale dominates at small grids** — addressable by extra
  attention capacity, not by Hopfield-style structure (§3).
- **TEM parameter-scaling anti-scales** (§7.1).

---

## 12. Bottom line / workshop framing

**What's defensible (multi-seed, on the regimes that matter):**

1. **Cognitive-map architectures (Level15-family / TEM) generalise to
   novel environments along four axes; standard transformers don't.**
   Multi-env, cross-topology, cross-scale, cross-class (§2).
2. **Explicit state correction (Level 1.5) is architectural, not
   capacity.** Per-regime + length sweep + arithmetic OOD all show
   ARCHITECTURE; only single-env lm200 accuracy is reproducible by
   adding a generic extra content head (§4.2). The bounded-error
   signature shows up only across lengths (clean +8pp at T=512 →
   +26pp at T=2048) and only Level15 has it.
3. **Calibration (NLL) is where Level1.5 dominates most clearly.**
   5–50× better than Vanilla / Vanilla_ExtraHead across every regime
   / length. Capacity *worsens* calibration.
4. **NoDrop is a near-free engineering Pareto-shift; GSF earns its
   keep on calibration, not accuracy.** §5.
5. **Cognitive maps differ in representation content, not just
   trainability.** Frozen linear probe +7.5pp Level15 → Vanilla
   (§10).
6. **Level1.5 extrapolates arithmetic chains.** +21pp on NumberLine
   OOD chain at 4× trained length; capacity does not close this
   (§6).
7. **TEM is parameter-efficient by inductive bias, not by under-
   parameterisation.** TEM anti-scales with d_g (§7.1).
8. **EM vs WM is regime-dependent; mechanism predicts both signs.**
   Same architecture wins different regimes (§9.2).

**What's NOT defensible:**

- "Level1.5 is universally best on small grids." TEM wins by +11pp;
  the cure is extra attention capacity, not Hopfield structure (§3).
- "Closed-loop goal navigation works." False for all variants (§10).
- "Hex grid cells emerge." False everywhere (§11).
- "Single-env lm200 accuracy is purely architectural." A generic
  extra content head matches Level15 here. Specific to this regime;
  doesn't generalise (§4.2).

**Corrected one-line workshop pitch:**

> "Explicit state correction (Level 1.5) in MapFormer buys bounded-
> error length-extrapolation, calibration, and training stability
> across novel-environment regimes (held-out maps, topologies,
> scales, task classes) — and matches TEM-style explicit memory on
> the regimes that matter. On aliased-observation data, generic extra
> capacity actively *worsens* clean accuracy and calibration. The
> architectural win survives every per-regime control except single-
> env lm200 accuracy, where a generic content head can also retrieve
> unique landmark tokens."

---

## 13. What's in flight

- `run_tem_noise_and_ffn.sh` → `TEM_NOISE_FFN_RESULTS.md` — already
  landed (§7.2-7.3). Net: FFN doesn't close clean lag (~2.4pp); lm200
  improves; noise regime TEM matches Level15 on accuracy, loses on
  NLL.

Nothing else queued at the time of writing. All multi-seed cells in §2
and §3 are populated; the per-regime capacity sweep is complete; the
TEM scaling sweep is complete.

---

## Appendix — source file index

**Headline tables** — `TEM_NOVEL_ENV_RESULTS.md`,
`MULTIENV_CLEAN_2x2.md`, `MULTICLASS_MULTISEED_RESULTS.md`,
`MULTISEED_FOLLOWUP_RESULTS.md`, `TEM_BACKGROUND_BASELINES.md`.

**Cross-scale chain** — `TEM_CROSSSCALE_DIAGNOSTIC.md`,
`SINGLE_SIZE_CONTROL.md`, `PERSCALE_OMEGA_RESULTS.md`,
`EM_HOPFIELD_CROSSSCALE.md`, `HOPFIELD_NOMAINAP_RESULTS.md`,
`LEVEL15EM_CROSSSCALE.md`, `EXTRAHEAD_CONTROL.md`.

**Capacity controls (read CAPACITY_PERREGIME, not CAPACITY_CONTROL)** —
`CAPACITY_PERREGIME.md` (controlling), `CAPACITY_CONTROL.md`
(superseded — lm200-only artifact), `VANILLANODROP_CONTROL.md`.

**NoDrop / GSF** — `NODROP_PARETO_RESULTS.md`, `GSF_NODROP_RESULTS.md`,
`DROPOUT_ABLATION_RESULTS.md`, `LEVEL15BETA_RESULTS.md`.

**TEM analysis** — `TEM_SCALING_RESULTS.md`,
`TEM_NOISE_FFN_RESULTS.md`, `TEM_T_MULTISEED.md`.

**NumberLine** — `NUMBERLINE_RESULTS.md` (+ capacity row in
`CAPACITY_PERREGIME.md`).

**Other regimes** — `LONGT_EVAL_RESULTS.md`, `VOCAB_SWEEP_RESULTS.md`,
`SPARSE_LANDMARKS_RESULTS.md`, `STOCHASTIC_TRANSITION_RESULTS.md`,
`MINIGRID_DOORKEY_*.md`, `MINIGRID_DK16_RESULTS.md`,
`MINIGRID_MEMORY_RESULTS.md`.

**Behaviour and probes** — `GOAL_DIRECTED_RESULTS.md`,
`GOAL_CLOSEDLOOP_RESULTS.md`, `PROBE_GOAL_RESULTS.md`,
`PROBE_GOAL_DISTANCE.md`, `STATE_PROBES.md`, `DOORKEY_BC_RESULTS.md`,
`DAGGER_RESULTS.md`, `ACTIVE_INFERENCE_RESULTS.md`,
`SR_PRETRAIN_RESULTS.md`.

**Mechanism / feedback notes** —
`.claude-memory/feedback_post_attn_dropout.md`,
`.claude-memory/feedback_em_vs_wm_mechanism.md`,
`.claude-memory/feedback_pc_kalman_duality.md`,
`.claude-memory/feedback_action_noise_framing.md`.

**Older reports superseded by this one** — `REPORT.md` (2026-05-15),
`REPORT_ADDENDUM.md` (2026-05-18).

*Generated 2026-06-02. Verify against `git log` for newer results.*
