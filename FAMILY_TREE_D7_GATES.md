# Family-tree task -- pre-flight gates (CPU, no training)

Ancestor tree depth 7 (255 nodes), 8 observation types, 8 relational actions.
Scored at REVISITED nodes. **chance = 1/8 = 0.1250.**

| baseline | value | want |
|---|---|---|
| chance | 0.1250 | — |
| marginal (most common observation) | 0.1290 | ~chance |
| **hub node** (always answer the most-visited node) | **0.1442** | ~chance |
| last scored observation | 0.1581 | ~chance |
| n-gram o1 / o2 / o3 / o5 | 0.1285 / 0.1238 / 0.1265 / 0.1294 | ~chance |
| oracle | 1.0000 | 1.0 |

## Is the task well-posed?

- revisit rate (scoreable steps): **0.215**
- visit concentration: top-1 node **0.021**, top-5 **0.069** of scored events
- **non-commutativity: 1.000** — fraction of nodes where mother-then-father != father-then-mother. Must be high, else the structure is effectively commutative and the task tests nothing MapFormer's SO(2) machinery cannot already do.

