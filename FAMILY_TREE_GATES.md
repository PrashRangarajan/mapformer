# Family-tree task -- pre-flight gates (CPU, no training)

Ancestor tree depth 5 (63 nodes), 8 observation types, 8 relational actions.
Scored at REVISITED nodes. **chance = 1/8 = 0.1250.**

| baseline | value | want |
|---|---|---|
| chance | 0.1250 | — |
| marginal (most common observation) | 0.1280 | ~chance |
| **hub node** (always answer the most-visited node) | **0.1628** | ~chance |
| last scored observation | 0.1580 | ~chance |
| n-gram o1 / o2 / o3 / o5 | 0.1235 / 0.1202 / 0.1241 / 0.1283 | ~chance |
| oracle | 1.0000 | 1.0 |

## Is the task well-posed?

- revisit rate (scoreable steps): **0.195**
- visit concentration: top-1 node **0.039**, top-5 **0.154** of scored events
- **non-commutativity: 1.000** — fraction of nodes where mother-then-father != father-then-mother. Must be high, else the structure is effectively commutative and the task tests nothing MapFormer's SO(2) machinery cannot already do.


## Verdict: PASS, with a raised floor

One shortcut was found and fixed before any training. Relations invert
(mother-then-child is the identity), so the walk oscillates and the same node's
observation repeats back to back:

| baseline | before dedup | after dedup | chance |
|---|---|---|---|
| n-gram order 2 | **0.3333** | **0.1202** | 0.1250 |
| n-gram order 3 | 0.3217 | 0.1241 | 0.1250 |
| last scored observation | 0.1916 | 0.1580 | 0.1250 |
| revisit rate | 0.748 | 0.195 | — |

Fix: score each node at most ONCE per episode.

**A second bug, in the gate itself.** The first version of this validator
reimplemented the walk inline instead of calling `env.generate_episode`, so the
dedup fix left every gate number unchanged -- the gate was testing a different
task from the trainer. It now calls the environment. A gate that duplicates task
logic can silently drift away from the thing it is meant to certify.

### The effective floor is 0.163, not 0.125

The hub-node baseline (always answer with the most-visited node's observation)
scores **0.163**, and last-observation **0.158**, both above the 0.125 chance
rate. This is real structure, not a bug: shallow nodes near ego are revisited far
more often, so their observations dominate the scored events. **Report model
accuracy against 0.163.**

### The task is well-posed

- **non-commutativity 1.000** — mother-then-father and father-then-mother land on
  different people for every node. No 2-D translation group can represent this.
- visit concentration is low (top-1 node 0.039, top-5 0.154), so no single person
  dominates.
- revisit rate 0.195 — about 1 in 5 steps is scoreable.
