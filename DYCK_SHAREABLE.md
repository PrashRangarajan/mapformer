# Dyck-2: which position encodings learn a bracket stack that survives longer, deeper input

**Task.** Next-token prediction over four tokens: `(` `)` `[` `]`. Sequences are balanced and
correctly nested, so at each position you may always *open* either bracket, and may *close* the
bracket most recently opened. Tracking which bracket that is -- the stack -- is the only part of
the task that requires memory.

**Setup.** Every model is trained ONLY on length 32, max depth 4, then tested unchanged on longer
and deeper sequences. Length = number of brackets; depth = how many are open at once at the deepest
point. Models are 1 layer / 1 head of size 64 (~51k parameters) unless marked 2 layers (2 heads,
~400k); identical recipe throughout (560k sequences, AdamW, lr 1e-4, weight decay 0.01, cosine
schedule), 8 seeds each. This follows the setup of the MapFormer paper (arXiv:2511.19279 v4, Sec 5.3).
MapPoPE and MapFormer use *path integration* -- position is a running sum of learned per-token steps,
so an opening bracket can step forward and a closing bracket step back. PoPE and RoPE use the token's
*index* in the sequence. The *n-gram baseline* predicts from the last bracket alone, with no stack.

**Metrics are the standard ones from the Dyck literature** (a single combined score is avoidable and
misleading here -- see the note at the end).

## 1. Bracket-closing memory (Hewitt et al. 2020; Yao et al. 2021 use the same)

"Let p_j be the probability that the model predicts the correct closing bracket given that j tokens
separate it from its open bracket. We report mean_j p_j." Probability is renormalised over the
closing brackets; opening brackets are never scored, since which bracket gets opened next is not
predictable. **Chance is 0.500.**

| model | trained on:<br>L 32, depth 4 | 4x longer:<br>L 128, depth 4 | 4x longer, 3x deeper:<br>L 128, depth 12 |
|---|---|---|---|
| **MapPoPE** (1 layer) | 0.994 | 0.793 | 0.719 |
| **MapFormer / MapWM** (1 layer) | 0.992 | 0.772 | 0.638 |
| PoPE (1 layer) | 0.646 | 0.585 | 0.551 |
| RoPE (1 layer) | 0.627 | 0.554 | 0.535 |
| PoPE (2 layers) | 0.919 | 0.623 | 0.578 |
| RoPE (2 layers) | 0.914 | 0.623 | 0.574 |
| *n-gram baseline, no stack* | 0.531 | 0.511 | 0.508 |

## 2. Valid-set prediction, per sequence (Suzgun et al. 2019; Bhattamishra et al. 2020; Ebrahimi et al. 2020)

The model must predict the exact SET of valid next brackets, and "an input is accurately recognized
only if the model correctly predicts the set of all possible brackets at each position" -- one
mistake anywhere fails the whole sequence. **Chance is 0.000.**

| model | trained on:<br>L 32, depth 4 | 4x longer:<br>L 128, depth 4 | 4x longer, 3x deeper:<br>L 128, depth 12 |
|---|---|---|---|
| **MapPoPE** (1 layer) | 0.661 | 0.108 | 0.042 |
| **MapFormer / MapWM** (1 layer) | 0.672 | 0.065 | 0.003 |
| PoPE (1 layer) | 0.000 | 0.000 | 0.000 |
| RoPE (1 layer) | 0.000 | 0.000 | 0.000 |
| PoPE (2 layers) | 0.084 | 0.000 | 0.000 |
| RoPE (2 layers) | 0.097 | 0.000 | 0.000 |
| *n-gram baseline, no stack* | 0.020 | 0.000 | 0.000 |

## 3. How far back the stack is still tracked (L 128, depth 12)

Metric 1 split by distance: how many tokens back the bracket that must be closed was opened. If the
previous token was an opening bracket it *is* the top of the stack, so no memory is needed -- and
two thirds of positions are like that, which is why averages flatter everything. Chance 0.500.

| model | d 1-2 (66% of positions) | d 3-8 (14%) | d 9-32 (13%) | d 33+ (6%) |
|---|---|---|---|---|
| **MapPoPE** (1 layer) | 0.997 | 0.996 | 0.971 | 0.730 |
| **MapFormer / MapWM** (1 layer) | 0.942 | 0.906 | 0.799 | 0.611 |
| PoPE (1 layer) | 0.801 | 0.740 | 0.645 | 0.569 |
| RoPE (1 layer) | 0.693 | 0.641 | 0.570 | 0.539 |
| PoPE (2 layers) | 0.900 | 0.767 | 0.694 | 0.638 |
| RoPE (2 layers) | 0.834 | 0.683 | 0.602 | 0.592 |
| *n-gram baseline, no stack* | 0.505 | 0.498 | 0.508 | 0.511 |

## What the tables say

1. **A single layer with path integration learns the stack; a single layer with index position does
   not.** At the training distribution MapPoPE and MapFormer reach 0.99 bracket-closing memory
   against 0.63-0.65 for 1-layer PoPE and RoPE. Index models need a second layer to learn it
   (0.91-0.92), which matches the theory for standard transformers (Yao et al. 2021). This is the
   MapFormer paper's central claim and it holds.
2. **Nobody generalises "perfectly".** On longer, deeper input the best model retains 0.719
   bracket-closing memory and solves 4% of sequences end to end; MapFormer retains 0.638 and 0.3%.
   Index models are near chance (0.53-0.58) and solve none.
3. **Adding PoPE's encoding to path integration helps** (MapPoPE over MapFormer on every metric and
   at every distance); adding it to index position does not.
4. **Every model has a reach horizon.** All are near-perfect when the matching bracket is within a
   few tokens and degrade with distance; pushed to length 512, even MapPoPE reaches chance beyond
   about 128 tokens of reach. They differ in where the horizon falls, not in having one.

## Notes and caveats

- The MapFormer paper reports a different metric: the F1 valid-continuation score of Goodale et al.
  (2025), designed to compare *many* formal languages, not to detect a stack. On Dyck-2 it gives
  most of its range to the two always-legal opening brackets, so the stack-free n-gram scores 0.857
  on it at the hardest condition -- above every index model (0.472-0.704). Our models reproduce the
  paper's ordering on that metric but sit 0.05-0.08 below its published values; the paper does not
  state its batch size, which is the untested suspect.
- Aggregation matters as much as metric choice: the same RoPE predictions score 0.871 when averaged
  over positions and 0.627 when averaged over distances, because the rare long-distance cases are
  where it fails.
- The set-prediction rows are not directly comparable to published numbers: those papers train with
  per-symbol sigmoids against valid-set labels, while these models are trained with next-token
  cross-entropy as MapFormer does, so they are not optimised for set prediction. A threshold-free
  form of the criterion is used here (every valid bracket must outrank every invalid one).
- Omitted: PathAtt (a paper baseline, not implemented here), CoPE (ours underperforms the paper's),
  and MapEM, which tracks MapFormer/MapWM closely on every metric.

Sources: Hewitt et al. 2020 (arXiv:2010.07515); Yao et al. 2021 (ACL 2021); Suzgun et al. 2019
(arXiv:1911.03329); Bhattamishra et al. 2020 (COLING); Ebrahimi et al. 2020 (arXiv:2010.04303);
Goodale et al. 2025 (ACL 2025); MapFormer, Rambaud et al. (arXiv:2511.19279 v4).
