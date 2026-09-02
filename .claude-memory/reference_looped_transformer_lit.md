---
name: reference-looped-transformer-lit
description: Mixture-of-Recursions and the recursive-transformer line — what our looped results do and do NOT claim novelty over.
metadata:
  type: reference
---

**Mixture-of-Recursions (MoR)**, arXiv 2507.10524, Bae, Y. Kim, Bayat, S. Kim, Ha,
Schuster, Fisch, Harutyunyan, Ji, Courville, Yun. v1 Jul 2025, v3 Oct 2025,
NeurIPS 2025 poster. Code: github.com/raymin0223/mixture_of_recursions.
Shared layer stack reused across recursion steps (parameter efficiency) PLUS
lightweight routers assigning a different recursion depth to each TOKEN, plus
selective KV caching for routed tokens. Claims a new Pareto frontier at equal
training FLOPs, 135M-1.7B. **Language modelling only** (val perplexity, few-shot).

**Consequence for our positioning: "recursion substitutes for depth" is NOT ours.**
That is the premise of the whole Recursive-Transformer/MoR line. Our index-arm
result (+0.363, horizon 9-16 -> 17-32, indistinguishable from 4 real layers at a
quarter of the parameters) REPLICATES a known result on a new task. Say so.

What has no counterpart on their side, and is where any novelty claim belongs:
- loop x PATH INTEGRATION composing super-additively on Match-Query (interaction
  +0.315, MDE 0.281) -- see [[project-loop-and-correction]]
- the loop's benefit being to the FLOOR (reliability/convergence), not the ceiling

**MoR is the principled version of our LoopedSampled**, which samples ONE global
count per training batch; MoR learns a per-token router. But two things say the
headroom here is small: LoopedSampled already scores 0.998 at ONE pass and
flattened the count-vs-accuracy curve from 0.178 spread to 0.001, and our depth
signal is SEQUENCE LENGTH, not token difficulty -- same weights, T=128 peaks at 4
passes and T=512 at 2. MoR routes on the token; nothing in it routes on length.
That direction tension is worth stating if the loop work is ever written up:
their premise is "hard tokens deserve more compute", ours is "more passes HURT
out of distribution".

Not verified here: Mixture-of-Depths (Raposo et al., DeepMind) is the other
"Mixture-of" in this neighbourhood -- token routing that SKIPS blocks in a
non-shared transformer, i.e. not weight-tied recursion. Check before citing.
