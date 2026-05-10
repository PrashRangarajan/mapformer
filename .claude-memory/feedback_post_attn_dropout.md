---
name: Post-attention dropout is harmful for landmark retrieval
description: The MapFormer baseline inherits Vaswani et al.'s post-attention residual dropout, which actively destroys rare-token retrieval. Removing it gives the bulk of TEM's lm200 lead. Use this finding before claiming β / sharper-softmax / multimodal-Bayes explanations.
type: feedback
originSessionId: continued-2026-05-10
---

**Discovered 2026-05-10.** Level15Beta (Level 1.5 + learnable softmax temperature β) closed +12pp of the Level15 → TEMFaithful gap on lm200 OOD T=512 (0.819 → 0.935). The natural interpretation was "sharper softmax helps landmark retrieval." That interpretation is **wrong**.

Two architectural differences between `WMTransformerLayer` and `WMTransformerLayer_Beta`:
1. β: learnable temperature on Q·K^T (init at 1/√d_head, the default scaling).
2. The post-attention residual add: original wraps `o_proj(out)` in `self.dropout`; Beta version drops the wrapper.

Learned β values barely moved from init: [0.148, 0.182] vs init 0.125. A 1.2–1.5× sharpening cannot plausibly explain +12pp.

**Ablation (Level15NoDrop, fixed β = 1/√d_head, only dropout removed):**

| Variant | lm200 OOD T=512 acc | NLL |
|---|---|---|
| Level15 | 0.819 ± 0.025 | 0.897 |
| Level15NoDrop | **0.948 ± 0.025** | 0.317 |
| Level15Beta | 0.935 ± 0.032 | 0.392 |
| TEMFaithful | 0.969 ± 0.010 | 0.171 |

NoDrop is essentially identical to Beta (NoDrop marginally *better*). Confirmed: removing the post-attn residual dropout is what unlocked the gap. β was a red herring.

**Pareto-trade-off (regime-dependent):**

| Config | T=512 acc Δ vs Level15 | T=512 NLL Δ |
|---|---|---|
| Clean | −0.7pp (within std) | **+100%** (NLL doubles — calibration loss) |
| Noise | +2pp | −6% |
| LM200 | **+12pp** | **−56%** |

Dropout regularisation helps when retrievals are *redundant* (aliased obs has ~128 copies per token; the random feature-zeroing averages out). It hurts when retrievals are *rare* (a landmark token appears once; zeroing 10% of its features destroys the signal).

**Implications:**
- Don't blame "sharper softmax" / "modern Hopfield β" / multimodal-Bayes for similar wins until you've ruled out architectural details like dropout, layer norm placement, residual wiring.
- The mechanistic story for TEMFaithful's lm200 lead may also be *partly* architectural-detail, not Bayesian-richness.
- Recommending "remove post-attn dropout" as a default change requires verifying clean NLL isn't a deal-breaker on the user's target regime.

**For the paper:** frame as "Vaswani et al.'s default block dropout is mistuned for cognitive-map tasks; a Pareto-shift toward landmark/noise regimes is achievable by removing one residual dropout, with a small clean-task NLL cost." Don't claim it as a strict improvement.
