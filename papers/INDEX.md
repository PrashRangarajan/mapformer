# Paper corpus

Every source `mapformer_math.tex` cites, stored so a claim can be checked without
a web round-trip. **`txt/` is tracked; `pdf/` is gitignored** (69 MB) — run
`bash papers/fetch.sh` to restore the PDFs, or `bash papers/fetch.sh <key>` for one.

Grep the corpus, don't re-search the web:

```bash
grep -n -i "data-dependent" papers/txt/hgrn.txt
```

## Verification status

Every row below was **read first-hand on 2026-09-06** unless marked otherwise.
The "checked" column names the specific claim in `mapformer_math.tex` §7 that the
reading confirms or corrects, with the line in `txt/` it was confirmed against.

### The frame itself

| key | arXiv / venue | checked |
|---|---|---|
| `puranik_janestreet` | Jane Street Blog, 22 Apr 2026 | **The frame is his.** linear + translation-invariant + continuous ⇒ one-parameter group ⇒ `A(d)=exp(dM)`; diagonalise ⇒ 1-D real blocks are exponential decay (NoPE at λ=0), 2-D conjugate pairs are `e^{λd}·rot(ωd)` = damped RoPE. Defective generators ⇒ polynomial terms, which he flags as unexplored and unexplained. Also: gated models should be read as **"learning how far to advance time"**, not as changing a decay rate — the reparameterised-time reading, published. |
| `grape` | 2512.07805, ICLR 2026 | The two-slot unification; App. D calls its own multiplicative extension "non-contextual"; App. E indexes the joint action by the offset; content-dependence requires `ω = g(x) ≥ 0`. |
| `alg_pe` | 2312.16045, NeurIPS 2024 | Maps an algebraic domain spec (sequences, **grids**, trees) to orthogonal operators. Structure-driven, not content-driven. |
| `jordan_rope` | 2605.04217, May 2026 | Fills Puranik's defective-generator gap: complex eigenvalue **coupled to a nilpotent in the same block**, giving modes `d^r e^{iωd}`. Not "unipotent" — non-semisimple. Needs a **contragredient query action** because the blocks are non-orthogonal (the same q/k asymmetry §3 derives). |
| `pj_rope` | 2606.05345, Jun 2026 | Fourier-jet-affine position space. Adjacent; not read in depth. |

### Content-dependent phase — the cell MapFormer is in

| key | arXiv | checked |
|---|---|---|
| `mapformer` | 2511.19279 | the reproduction target |
| `srope` | 2511.17388, ICLR 2026 | `θ = τ·cumsum(σ(W_g x) ⊙ (κ * W_ω x))` |
| `carope` | 2507.23083, Jul 2025 | **`f(x_t) = 1/(softplus(x_t W)+1) ∈ (0,1)`**, accumulated. Strictly positive AND bounded above: it cannot reverse *or stay*. `W ∈ R^{d×h}` is **one scalar per head** — rank 1 per head, below MapFormer's `r`. Two axes differ, not one. |
| `mamba3` | 2603.15569 | `Diag(A(t) + iθ(t))`, both parts data-dependent; Prop. 3 "Complex SSM, Data-Dependent RoPE Equivalence". |
| `liere` | 2406.10322, ICML 2025 | `R(p) = exp(Σ_i p_i A_i)`, `{A_i}` a **learned skew-symmetric basis** — structurally MapFormer's `ActionToLieAlgebra` driven by a known **position** instead of by content. Fig. 6 sweeps generator tile size 2×2…48×48 and **peaks at 8×8**, an interior optimum. The nearest published neighbour to our rank axis; different axis (output-side density, not input-side bottleneck) and the same shape of finding. |

### Content-dependent magnitude

| key | arXiv | checked |
|---|---|---|
| `pope` | 2509.10534 | Source of the polar frame equation. `μ = softplus(·) ≥ 0`. PoPE is defined by **deleting** RoPE's implicit content-phase interaction `φ_k − φ_q` — the exact complement of MapFormer, which amplifies it. |
| `fox` | 2503.02130 | `f_t = σ(w_f^T x_t + b_f)` scalar per head; log-cumsum bias on the logit. |
| `gla` | 2312.06635 | `S_t = Diag(α_t) S_{t-1} + k_t v_t`, α from a sigmoid. |
| `mamba`, `mamba2` | 2312.00752, 2405.21060 | selective diagonal / scalar-times-identity `A`. |
| `hgrn` | 2311.04823, NeurIPS 2023 | **Table 11 ablates a data-dependent phase and finds it WORSE**: "the experiments show that the phase argument θ should not be data-dependent". Published negative for content-dependent phase on language. |
| `hgrn2` | 2404.07904 | Table 1: real HGRN at 2× state size beats complex HGRN — the complex-recurrence gain was **state expansion**, not the phase. |

### Index-driven (the classical rows)

| key | arXiv | checked |
|---|---|---|
| `rope` | 2104.09864 | — |
| `alibi` | 2108.12409 | linear-in-lag logit penalty; Puranik shows it is realisable by a defective generator. |
| `xpos` | 2212.10554 | rotation **and** exponential decay, index-driven. |
| `nope` | 2305.19466 | — |

### Out of the frame

| key | arXiv | what it breaks |
|---|---|---|
| `path` | 2505.16381 | non-abelian: cumulative **product** of Householders, each `I + rank-one`, data-dependent. No `S_t`. |
| `deltanet` | 2406.06484 | generalized Householder recurrence; WY representation. |
| `rwkv7` | 2503.14456 | delta rule with vector-valued gating. |
| `cope` | 2405.18719 | `g_ij = σ(q_i·k_j)`, `p_ij = Σ_{j<k≤i} g_ik` — position depends on the **query–key pair**, so no per-token `θ_t` exists. Sigmoid ⇒ also a monotone clock. |
| `tape` | 2501.00712 | **CORRECTION:** written "TAPA" and grouped with CoPE in an earlier draft. It is TAPE, and its driver is not a pair — it updates a positional tensor **across layers** through attention and MLP layers under an equivariance constraint. What it breaks is the **path structure** (A8): the update is not an accumulation of per-token increments, so it belongs with MesaNet/Titans. |
| `mesanet` | 2506.05233 | locally optimal least squares by conjugate gradient — a prefix-global inverse. |
| `titans` | 2501.00663 | deep neural memory updated by gradient descent with momentum. |

## What the reading changed

1. **The frame is Puranik's and GRAPE's**, and Puranik gives the reason it is
   exhaustive (one-parameter group + Jordan normal form) that §1–2 only asserted.
2. **CARoPE differs from MapFormer on two axes, not one** — sign *and* rank.
3. **Two published negatives for content-dependent phase on language** (HGRN
   Table 11, HGRN2 Table 1), independent of our own underpowered enwik8 result.
   All three point the same way, and none of them touches navigation.
4. **LieRE is the rank axis's nearest neighbour** and must be cited: a learned Lie
   generator with a density sweep that peaks in the interior.
5. Three table errors: TAPE's name and group, Jordan-RoPE's algebra, CARoPE's rank.
