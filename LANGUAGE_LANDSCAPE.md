# Language tasks: what is already done, and what is actually open

Literature check 2026-08-27, before spending GPU. **Headline: the obvious language
experiments are all done, three times over. Do not re-run them.** The open slots are
methodological and benchmark-shaped, not mechanism-shaped.

## 0. THE MAPFORMER PAPER IS NOW AT v4 AND ALREADY DID LANGUAGE

CLAUDE.md is written against **v1 (Nov 2025)**. The paper is at **v4 (10 May 2026)**;
**§5.5 "Scalability to Natural Language" and Appendix B.5 are NEW**. Verbatim:

> "we pretrained a 12-layer MapWM on OpenWebText against a RoPE baseline, using the
> nanoGPT codebase on 4 H100 GPUs for approximately 10^11 tokens. Overall, MapWM's
> training remained stable and yielded a consistent perplexity improvement over RoPE
> (**RoPE 19.14+/-0.14 vs MapWM 18.79+/-0.15**) and better length extrapolation...
> However, matching the length-extrapolation of 1D path-integrating baselines like
> **CoPE and PathAtt** would likely require more hyperparameter tuning."

Setup: 12 layers, embed 768, head 64, rank r=4, base freq 1024, ctx 1024, 5 seeds,
p<0.005.

Two honest negatives THEY report:
- **BLiMP 0.78+/-0.03 (RoPE) vs 0.79+/-0.02 (MapWM) -- no gain.** "the observed gains
  on OpenWebText **do not come from better syntactic modeling**, which might require
  another mechanism than (commutative) path-integration."
- **Length extrapolation LOSES**: on NarrativeQA "its perplexity still degrades
  sharply compared to the numbers reported for CoPE and PathAtt."

They also nominate our best remaining target: "**code modeling, where deep nested
structures are more common, could be a natural setting to test this further.**"

**The paper itself frames MapFormer-on-language as the same family as CoPE/PaTH.**

## 1. PoPE is NOT content-dependent (correction to an assumption we made)

PoPE (arXiv:2509.10534, **ICML 2026**, Gopalakrishnan/Csordas/Schmidhuber/Mozer).
Its *magnitude* is content-dependent (softplus); its **rotation ANGLE is purely
position-dependent** (phi_q = t*theta_c, phi_k = s*theta_c). PoPE is the OPPOSITE of
a content-dependent position -- it is not in the CoPE family at all.

Language, OpenWebText, GPT-2 tok, ~9B tokens, seq 1024, RoPE-only baseline:

| params | RoPE ppl | PoPE ppl |
|---|---|---|
| 124M | 21.55 | 21.33 |
| 253M | 18.88 | 18.55 |
| 774M | 15.85 | 15.45 |

Length-extrapolation claim (PG-19 to 10,240) has **NO numeric table -- figure only**.
No independent replication exists.

## 2. CoPE: marginal on prose, categorical on structure

CoPE (arXiv:2405.18719, Golovneva et al.). Mechanism: gate `g_ij = sigmoid(q_i.k_j)`
(**query-dependent**), `p_ij = sum_{k=j}^{i} g_ik`, interpolate learned embeddings at
fractional p, add as a **logit bias** (not a rotation). Positions capped p_max=64.
`p_ij` is an **O(T^2)** matrix -- the cumsum is over the full attention matrix, NOT a
linear-time scan.

| Wikitext-103 (124M, ctx 1024) | test ppl | | algorithmic (error %) | RoPE | CoPE |
|---|---|---|---|---|---|
| Absolute PE | 24.87 | | Flip-Flop OOD | 20.3 | **4.9** |
| Relative PE | 23.81 | | Selective copy OOD | 100.0 | **0.0** |
| CoPE | 23.46 | | Counting, 3 var | 17.8 | **1.2** |
| CoPE + Relative | **23.23** | | Counting OOD | 34.1 | **4.0** |

Code (20M, ctx 4096): RoPE 4.1 -> CoPE 3.9. **~1.5% on plain LM; categorical on
structured recurrence.** Not composable with everything: CoPE_ALiBi is WORSE than
plain absolute PE.

## 3. THE MECHANISM GAP IS CLOSED -- Selective RoPE is our exact construction

**Selective RoPE (arXiv:2511.17388, ICLR 2026)**, their Figure 4 verbatim:

```python
omega = W_omega @ q ; omega = conv1d(omega)
omega = temp * cumsum(omega)
sin_o, cos_o = sincos(omega) ; return rope(q, k, cos_o, sin_o)
```

That is `theta = omega * cumsum(Delta(token))` applied as a RoPE rotation, on language.
Differences from MapFormer: projects from q (not a rank-2 bottleneck on the embedding),
adds a sigmoid phase gate, learnable temperature instead of a fixed geometric omega
ladder. Framed as `S_t = S_{t-1} R_t + v_t k_t^T` -- input-dependent rotation as a state
transition, i.e. path integration.

Honest, modest results (FineWeb, ctx 4096): GLA 1.3B Wiki ppl RoPE 18.50 -> SRoPE
17.87, avg acc 54.4 -> 54.6. **FoX 370M REGRESSES badly** (Wiki 25.29 -> 33.87,
unexplained). Reports **training instabilities at higher LR**. Wins decisively on
parity/state-tracking and copying-with-length-extrapolation. Does NOT cite CoPE or PaTH.

**PaTH (arXiv:2505.16381, NeurIPS 2025)** generalises further: cumulative PRODUCT of
data-dependent Householder transforms `H_t = I - beta_t w_t w_t^T`, deliberately
non-commutative and non-diagonalizable. Strongest numbers in the family (760M, 50B tok):
WikiText 19.01 -> 18.03, LAMBADA 19.77 -> 16.79, **RULER 16K: RoPE 0.0 -> PaTH 18.7**.
PaTH's critique of the CoPE family: "these approaches operate solely at the attention
logit level... **the dot-product structure is fundamentally limited**".

Also close: CARoPE (input-dependent frequencies, thin paper), APE (cumulative
orthogonal products but data-INdependent), LieRE (Lie-group but coordinate-driven,
vision), DAPE (MLP on attention scores, logit-level).

## 4. MapFormer vs CoPE -- the three real differences

1. **Query-independent vs query-dependent (the big one).** MapFormer's
   `Delta_t = W_out W_in x_t` depends only on token t, so `theta_t` is a genuine
   STATE: one scalar per position, shared by all queries, computable by prefix sum
   -> **O(T)** and parallel-scannable. CoPE's `g_ij` depends on both endpoints, so
   "distance" differs per query -> **O(T^2)**, no scan.
2. **Rotation vs additive logit bias.** MapFormer applies `R(theta_i - theta_j)` to
   q/k -- position enters multiplicatively, inheriting RoPE's relative-shift property.
   CoPE adds a bias after the dot product. This is exactly what PaTH criticises.
3. **Group structure.** MapFormer is a Lie-group construction, extends to nD and
   non-commutative groups (MapEM-NC). CoPE is strictly scalar/1D.

## 5. Overclaiming flags for this subfield

1. **"Strong length extrapolation" is the most abused phrase here.** PoPE claims 10x
   with no table. MapFormer measured its own and found it degrades vs CoPE/PaTH. Same
   family, opposite claims, because nobody runs head-to-head.
2. **Baseline-of-one everywhere.** PoPE compares only to RoPE. Selective RoPE cites
   neither CoPE nor PaTH.
3. **NoPE keeps winning and keeps getting buried.** In Selective RoPE's own table,
   plain NoPE has the best avg accuracy at GLA 1.3B (55.2). Discount any result here
   lacking a NoPE arm.
4. **Perplexity and capability dissociate.** Plain-LM gains are 1-3% and do not
   transfer; structured/algorithmic gains are categorical. MapFormer wins OpenWebText
   ppl and TIES BLiMP.
5. **No independent replications exist** for CoPE, PoPE, PaTH, or Selective RoPE.
   All author-reported.

## 6. What is ACTUALLY open (ranked)

**(a) Distance-stratified bracket/scope matching on real code. -- RECOMMENDED.**
No benchmark exists. Existing code benchmarks (LCC/LongCoder, DI-Bench, CoRe,
Code2Bench) measure completion or dependency inference, not PE behaviour at
CONTROLLED opener-to-closer distance. CoPE's code experiment reports aggregate ppl
only. **The MapFormer authors explicitly nominate this.** And it is OUR crossover
experiment in language form: our result says content-dependent position wins when
recurrence distance exceeds attention's horizon; stratifying by bracket distance
measures exactly that curve. We already have Dyck + Match-Query infrastructure and
the gates (measured floor, n-gram, context-destruction) this subfield lacks entirely.

**(b) A matched-compute head-to-head** of MapWM / CoPE / PaTH / Selective RoPE /
NoPE / RoPE. Every paper uses a different backbone, dataset and token budget; no two
are comparable. **Our standing rule 3 is precisely the methodological gap here.**
Expensive but high-value to the field.

**(c) The commutativity question.** MapFormer's own stated limitation ("unclear how
limiting the commutative constraint will prove for language"), testable against PaTH
(non-commutative by construction) and our existing MapEM-NC arm. Tempered by our own
family-tree result: non-commutativity bought +0.014 for 34x the cost.

**Do NOT run:** MapWM-vs-RoPE on OpenWebText (done, 5 seeds, 10^11 tokens);
PoPE on language (done, ICML 2026); content-dependent cumsum rotation on language
(done, Selective RoPE ICLR 2026).
