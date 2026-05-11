---
name: Multiple independent fixes each ~match TEMFaithful's lm200 lead
description: At least two architectural changes — removing the post-attn residual dropout, and switching to K=8 multi-modal Bayes (GSF) — each independently close ~14pp of the Level15 → TEMFaithful gap on lm200. Updates the framing of "what's special about TEMFaithful": it's a collection of design choices, several of which we can match without TEM machinery.
type: feedback
originSessionId: continued-2026-05-10
---

**Headline lm200 OOD T=512 numbers (n=3 each):**

| Variant | Architecture difference vs Level15 | lm200 acc | lm200 NLL |
|---|---|---|---|
| Level15 | (baseline) | 0.819 | 0.897 |
| Level15NoDrop | remove post-attn residual dropout | 0.948 | 0.317 |
| Level15GSF | K=8 parallel Kalman chains + mixture | 0.956 | 0.227 |
| Level15GSF_NoDrop | both fixes (queued 2026-05-10) | ? | ? |
| TEMFaithful | per-action W_a + Hopfield read | 0.969 | 0.171 |

Each fix in isolation closes ~14pp of the 15pp gap. We have at least TWO independent paths to ~TEMFaithful performance, neither of which uses TEM's actual machinery (per-action transition matrices, Hopfield memory).

**Implication for paper framing:**
- "Per-action W_a is essential for landmark retrieval" — FALSE. We can match it without W_a.
- "MapFormer fundamentally can't do landmarks" — FALSE. Two architectural tweaks to MapFormer each match TEMFaithful.
- "TEMFaithful wins for a single mechanistic reason" — UNLIKELY. Two unrelated changes (signal preservation through residual, multi-modal posterior) each capture most of the gap.

**Honest framing:** TEMFaithful's lm200 lead is probably a *combination* of design choices (some incidental like its lack of dropout, some intentional like its memory mechanism). MapFormer-with-Level15 can match it via different paths. The "TEM advantage" is mostly recoverable inside the MapFormer-WM architecture.

**Methodological caution:** when a baseline beats your model by a clear margin, the first instinct is "their mechanism is special." Often the gap is actually several small design choices stacked. Audit the diff carefully — we found the dropout difference by literally diffing transformer layer classes. β / "sharper softmax" was the obvious-looking culprit and turned out to be a red herring; the actual fix was a one-line dropout removal.
