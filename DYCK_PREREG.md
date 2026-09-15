# DYCK_PREREG -- replicating MapFormer v4's Dyck-2 result (Sec 5.3, Fig 3a-b, Fig 6b-f, App. B.4)

Written after the gates (`DYCK_GATES.md`) and a 500-step smoke test of each architecture, before
any full training run. Committed before launch.

## 1. What the paper claims (the replication targets)

Sec 5.3: "MapFormers with a single layer perfectly generalize OOD. In contrast, baseline models
require a second layer to approach near-perfect training performance and ... fail to generalize
beyond the training distribution." Fig 6 caption: "Only MapFormers reach perfect performances on
the training set (L 32, D 4) and maintain strong performances in the hardest OOD setting (L 128 D 12)."

Fig 6 F1 grids, read from the figure (rows D, columns L = 32/64/96/128):

| arm | D4 | D6 | D8 | D12 |
|---|---|---|---|---|
| RoPE-1L | .91 .84 .80 .78 | .89 .84 .80 .78 | .88 .83 .79 .78 | .86 .81 .78 .77 |
| RoPE-2L | .97 .70 .63 .59 | .89 .66 .60 .56 | .79 .61 .56 .53 | .66 .54 .52 .50 |
| PathAtt-1L | .95 .91 .88 .85 | .96 .92 .89 .87 | .95 .92 .90 .88 | .94 .91 .89 .88 |
| MapWM-1L | 1.00 .99 .98 .97 | .99 .97 .96 .96 | .98 .96 .95 .95 | .98 .94 .94 .94 |
| MapEM-1L | 1.00 1.00 .99 .99 | 1.00 .98 .98 .97 | .99 .97 .96 .96 | .98 .95 .95 .95 |

Fig 3a (D=4 row) and 3b (L=32 column) add CoPE, read off the plot (+/-0.01):
CoPE-1L L32..128 at D4 ~ .92 .86 .85 .84, D4..12 at L32 ~ .92 .93 .93 .93;
CoPE-2L ~ .96 .82 .79 .78 and .96 .90 .85 .82; PathAtt-2L ~ .97 .70 .60 .55 and .97 .91 .86 .83.

## 2. Protocol

**Stated by the paper and followed exactly**
- Dyck-2 over () and []; sequences of fixed length L whose maximum depth is D and which return to
  depth 0 at L; the sampling distribution is modified step by step to guarantee this.
- F1 valid-continuation metric of Goodale et al. (2025), exactly as printed in App. B.4.
- 1-layer models: 1 head, head size 64 (d_model 64). 2-layer models: 2 heads of size 64 (d_model 128).
- Train L=32, D=4, 560,000 sequences; AdamW, lr 1e-4, weight decay 0.01; cosine schedule with warmup.
- Eval grid L in {32,64,96,128} x D in {4,6,8,12}.
- MapWM rank r=2: Fig 3f plots the Dyck model's cumsum(Delta_in) "in R^2". MapEM is MapEM-os
  (separate q0/k0, observation and structure), the paper's headline EM.

**Not stated; chosen here**
| detail | choice | why |
|---|---|---|
| batch size | 128 | the paper's navigation batch; Dyck text says "like in the navigation task" |
| warmup / cosine floor | 5% linear, cosine to 10% | repo convention (nanoGPT, which the paper uses, also floors at 10%) |
| sampler "modification" | uniform open/close when both feasible, forced otherwise; bracket type uniform | minimal modification satisfying the stated constraints |
| start token | BOS | Fig 3f "starting in symbol x" |
| F1 aggregation | mean of per-prefix F1 over the L prefixes | the formula is per prefix s |
| training loss | next-token CE on every position | Goodale et al.'s setup; the paper trains everything by next-token prediction |
| dropout | 0.1 | the models' default, used in the navigation reproduction |
| omega base | 32 | App. A.8 sets base = the longest distance; the paper's only non-grid base is its context size |
| eval set | 1024 sequences per cell, fixed seed | |
| seeds | 8 per arm, one batch | paper shows bands, no count |

**Not replicated**: PathAtt (not built in this repo). CoPE is the repo's implementation.

## 3. Floors from the gates (the paper reports none)

At L32 D4 / L128 D4 / L32 D12 / L128 D12: best n-gram (orders 1-6, fitted on L32 D4) 0.880 / 0.873 /
0.903 / 0.884; trigram 0.846 / 0.832 / 0.903 / 0.884. The CE-optimal predictor (the sampler's own
distribution, exact zeros) scores 0.911 / 0.933 / 0.753 / 0.944: a model that fits the training
loss perfectly is NOT at F1 = 1, because opening at depth D is valid but never sampled.
Uniform over Val(s) scores 1.000 (metric check).

## 4. Registered verdicts

Replication tolerance for a paper value v: the arm's seed mean is within 0.03 of v.

- **R1** MapWM-1L and MapEM-1L at L32 D4: paper 1.00. REPLICATES if mean >= 0.97.
- **R2** MapWM-1L and MapEM-1L at L128 D12: paper 0.94 / 0.95. REPLICATES if mean >= 0.91 / 0.92.
- **R3** RoPE-2L reaches the training cell (paper 0.97; replicates if >= 0.94) AND collapses with
  length (paper L128 D4 0.59 and L128 D12 0.50; replicates if both <= 0.70).
- **R4** RoPE-1L does not reach the training cell (paper 0.91; replicates if < 0.94).
- **R5** CoPE-1L / CoPE-2L at L32 D4 and L128 D4 within 0.03 of Fig 3a (exploratory: values read off a plot).
- **R6** the headline contrasts at L128 D12, paired by seed, MDE = 2.8 sd / sqrt(8):
  MapWM-1L - RoPE-2L (paper +0.44), MapWM-1L - RoPE-1L (paper +0.17), MapEM-1L - RoPE-2L (+0.45).
  DETECTABLE if |delta| > MDE.
- **R7** (floor reading, not in the paper) whether each MapFormer arm exceeds the best n-gram floor
  at L128 D12 (0.884) by more than its MDE.
- Grid-wide: for every arm with a Fig 6 grid, report the max |ours - paper| over the 16 cells.

**Convergence (rule 10).** Report final-10% loss slope and final CE against the floor (0.800 at
L32 D4). If any MapFormer or RoPE arm is still descending (slope < -0.005 per 1k steps on the
median seed), the batch-size choice is suspect: a pre-registered follow-up reruns all arms at
batch 32 (17,500 steps, the same 560,000 sequences). Its verdicts are reported alongside, not
instead. **Rule 9**: r(final loss, F1) per cell.

**Mechanism (Fig 3f, eval-only, secondary).** For MapWM-1L, W_in applied to each bracket's
embedding: the paper says same-type open/close point opposite (cos ~ -1) and the two types use
orthogonal directions (|cos| ~ 0). Report both cosines per seed.

**Exploratory, cheap, not in the paper's figure**: MapWM-1L and MapEM-1L at r=4 (App. B.5 mentions
r=4 for depth); read as exploratory whatever they show.
