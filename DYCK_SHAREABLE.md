# Dyck-2: which position encodings keep tracking a bracket stack on longer, deeper input

**Task.** Next-token prediction over four tokens: `(` `)` `[` `]`. The sequences are balanced and
correctly nested, so at every position the legal next tokens are: either bracket may be *opened*
(always), and the bracket most recently opened may be *closed* (when something is open). Scoring
the exact next token is meaningless -- the sequences are sampled, so several continuations are
genuinely possible -- so we score the predicted distribution against the legal set.

**Training.** All models trained ONLY on length 32, max depth 4, then tested unchanged on longer and
deeper sequences. Length = number of brackets. Depth = how many are open at once at the deepest
point. 1 layer, 1 head of size 64, ~51k parameters each, identical recipe (560k sequences, AdamW,
lr 1e-4, weight decay 0.01, cosine schedule), 8 seeds. Setup follows the MapFormer paper
(arXiv:2511.19279 v4, Sec 5.3).

**Two metrics, kept separate.** They measure different failures and neither implies the other.
- **invalid mass** -- probability placed on brackets that are ungrammatical there. Lower is better;
  a uniform guesser is ~0.25. This is the grammaticality/coverage half.
- **stack accuracy** -- at positions where something is open, is P(legal closer) > P(illegal
  closer)? **Chance is 0.500.** This is the only part of the task that requires tracking the stack;
  the always-legal opening brackets cannot inflate it.

| | | trained on:<br>length 32, depth 4 | 4x longer:<br>length 128, depth 4 | 4x longer, 3x deeper:<br>length 128, depth 12 |
|---|---|---|---|---|
| **MapPoPE** | invalid mass<br>stack accuracy | 0.002<br>1.000 | 0.028<br>0.991 | **0.018**<br>**0.978** |
| **MapFormer (MapWM)** | invalid mass<br>stack accuracy | 0.003<br>0.999 | 0.047<br>0.965 | 0.054<br>0.928 |
| **PoPE** | invalid mass<br>stack accuracy | 0.078<br>0.940 | 0.217<br>0.904 | 0.195<br>0.864 |
| **RoPE** | invalid mass<br>stack accuracy | 0.078<br>0.936 | 0.241<br>0.851 | 0.220<br>0.822 |
| *n-gram baseline (no stack)* | invalid mass<br>stack accuracy | *0.125*<br>*0.797* | *0.130*<br>*0.791* | *0.116*<br>*0.770* |

Seed means, n=8; seed sd is <=0.006 on the training condition and <=0.036 elsewhere. MapPoPE and
MapFormer use path integration (position is a running sum of learned per-token steps, so an opening
bracket can step forward and a closing bracket step back); PoPE and RoPE use the token's index in
the sequence. The *n-gram baseline* predicts from the last bracket alone, with no stack.

**Stack accuracy averages over easy and hard positions -- split it by reach.** If the previous token
was an opening bracket, it *is* the top of the stack, so no memory is needed; two thirds of
positions are like that. The real measure is how far back the model can still find the top of the
stack. At length 128, depth 12, by distance to the bracket that must be closed:

| | 1-2 (66% of positions) | 3-8 (14%) | 9-32 (13%) | 33+ (6%) |
|---|---|---|---|---|
| MapPoPE | 0.999 | 0.996 | 0.971 | **0.730** |
| MapFormer (MapWM) | 0.988 | 0.906 | 0.799 | **0.611** |
| PoPE | 0.961 | 0.740 | 0.645 | 0.569 |
| RoPE | 0.937 | 0.641 | 0.570 | 0.539 |
| *n-gram baseline* | *0.904* | *0.498* | *0.508* | *0.511* |

**What this says.**
1. On the training condition every model is near-perfect on both metrics: all four reach 0.94-1.00
   stack accuracy. The differences are entirely out of distribution.
2. Out of distribution, path integration keeps the stack (0.93-0.98) while index position loses it
   (0.82-0.86) and also becomes ungrammatical, putting a fifth of its mass on illegal brackets.
3. The n-gram is at chance (0.50) past distance 2, as something with no stack must be -- but it is
   *more grammatical* than either index model out of distribution (0.116 vs 0.195-0.241 invalid
   mass). The two metrics genuinely dissociate.
4. Every model has a horizon. Push sequences to length 512 and even MapPoPE falls to chance beyond
   about 128 tokens of reach; it just gets further than the others.

**Caveats.** The paper reports a single combined F1 in which these two quantities are fused, and
which we also computed: on that metric our path-integration models land 0.05-0.08 below the paper's
published values at the hardest condition, so the ordering replicates but the levels do not (the
paper leaves batch size unspecified; that is the untested suspect). That combined F1 tracks invalid
mass almost perfectly and is nearly blind to stack accuracy -- the no-stack n-gram scores 0.857 on
it, above every index model. MapPoPE and PoPE are not in the paper. Two of the paper's baselines are
omitted: PathAtt (not implemented here) and CoPE (our implementation underperforms theirs).
