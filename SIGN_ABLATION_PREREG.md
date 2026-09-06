# Pre-registration: the sign ablation (axis A5)

Written 2026-09-06, **before** any arm was trained. Verdicts below are fixed in
advance; whichever way they point, that is what goes in the results file.

## The claim

The relational map has eight axes. Seven are varied deliberately somewhere in the
literature. **A5 — the sign of the accumulated phase increment — is not.** Every
content-dependent-phase mechanism outside MapFormer is non-negative, and in every
case that is a side-effect of a squashing function, never an argued design choice
(all four read first-hand, `papers/INDEX.md`):

| mechanism | increment | sign |
|---|---|---|
| CARoPE | `1/(softplus(xW)+1)` | `∈ (0,1)` — cannot reverse **or stay** |
| GRAPE-AP | `ω = g(x)` | `≥ 0`, required by the construction |
| CoPE | `σ(q_i·k_j)` | `≥ 0` |
| FoX | `log σ(·)` | `< 0` by construction |
| **MapFormer** | `W_out W_in x` | **signed**, because nothing squashes it |

**Mechanism.** Under non-negativity the per-channel angle
`θ_t^(c) = ω_c · Σ_{u≤t} Δ_u^(c)` is monotone in `t`. Monotone is all a language
clock needs. It is not enough for a cognitive map: on the torus east and west must
cancel, and a monotone code can encode `(n_E, n_W)` but not `n_E − n_W`. Revisit
retrieval needs the latter — two paths reaching the same cell by different step
counts must produce the *same* phase.

> **Prediction: invisible on language, decisive on navigation.**
> Only the navigation half is tested here. The language half is left alone
> deliberately: our enwik8 setup is underpowered by a factor of ~1 (MDE 0.0041
> against an expected 0.0067 bits/byte), and there are already two published
> negatives for content-dependent phase on text (HGRN Table 11, HGRN2 Table 1).

## Arms — 6 × 12 seeds = 72 runs, one batch

Clean torus paper task, `grid 64`, `T=128`, 1 layer, 2 heads, `d=128`,
300 epochs, warmup+cosine, **lr 1e-3** (recipe C1: cuts Vanilla's seed sd 3.5× at
T=512, which is what sets every MDE here). Held-out map, env-seed 10000.

| arm | Δ | role |
|---|---|---|
| `Signed_r4` | `W_out W_in x` | **the baseline for the primary contrast** |
| `Abs_r4` | `\|W_out W_in x\|` | **THE PRIMARY ARM** |
| `Pos_r4` | `softplus(·)` | GRAPE-AP style |
| `CARoPE_r4` | `1/(softplus(·)+1)` | CARoPE verbatim |
| `Vanilla_r4` | `W_out W_in x` | RNG/construction-path control |
| `RoPE` | index | the floor a monotone clock should approach |

Parameter count is **205,785 in all five MapFormer arms** — the constraint adds
nothing and does not touch the rank of the content-to-angle map.

**Why `Abs` is the primary and the other two are secondary.** `Abs` varies A5 and
nothing else: same parameters, same map, no squashing (gradient magnitude
preserved), `Δ` still reaches 0 exactly, and at init `|Δ|` is half-normal with the
same scale as the signed `Δ`. `Pos` and `CARoPE` carry an **init confound** —
`softplus(0)=0.693`, `1/(softplus(0)+1)=0.591`, so at initialisation every token
advances the clock by a constant and the model *is* RoPE, whereas the signed arms
start near `Δ=0`, i.e. near NoPE. For CARoPE that is faithful (its paper
initialises to RoPE on purpose), but it means a deficit in those two arms is not
attributable to sign alone.

**Why `Signed_r4` and not `Vanilla_r4` is the baseline.** The constrained arms
replace `action_to_lie` after construction, drawing a second time from the RNG.
`Signed_r4` takes the identical construction path with `mode="signed"` and is
**verified bit-identical to `Vanilla_r4` given the same weights (max|diff| = 0.0)**,
so it isolates the constraint from the init shift. `Signed_r4 − Vanilla_r4` is
reported as the control; if it is not inside its MDE, the batch has a problem and
the primary contrast is not readable.

## Verdicts, fixed in advance

Paired per-seed, at T = 128 / 512 / 1024. MDE = 2.8·sd/√12.

- **H1 (primary).** `Abs_r4 − Signed_r4`.
  - `< −MDE` at T=128 **and** at OOD lengths → **sign is load-bearing for
    navigation**, as predicted.
  - `< −MDE` at OOD lengths **only** → **NOT the predicted result.** This project's
    universal signature is "helps at OOD length" — rank, the InEKF, the forget
    gate and PoPE all show it and nothing explains it. An OOD-only sign effect is
    another instance of that unexplained axis, not evidence for the
    net-displacement mechanism, and must be reported as such.
  - `|Δ| < MDE` everywhere → **the mechanism claim is refuted at n=12.** A monotone
    clock does the torus task as well as a signed one, and A5 is not the opening
    the map suggested.
- **H2.** `Pos` and `CARoPE` should track `Abs` in direction, and should not fall
  below `RoPE` — a constrained MapFormer that is *worse* than an index model has
  broken rather than degraded, and would point at the init confound.
- **H3 (scope).** If H1 fires, the deficit should be largest where net displacement
  matters most. Reported, not pre-registered as a verdict: the T=128 → T=1024
  degradation slope per arm.

## Analysis plan

Rule 9 applies. `r(final loss, accuracy)` is computed per length and reported; the
**loss-matched residual contrast is primary** wherever `|r| > 0.5`, with the raw
contrast printed beside it. Both go in the file whichever way they point.

Rule 11 applies: any contrast inside its MDE is reported as **unmeasured**, never
as a null.

## Gates, run before the batch

1. Parameter count identical across all five MapFormer arms. **PASS** (205,785).
2. `Signed_r4` bit-identical to `Vanilla_r4` on shared weights. **PASS** (0.0).
3. `Δ ≥ 0` for every constrained arm at init **and after training**. Checked at
   init (PASS: min +2e-05 / +0.246 / +0.410); re-checked post-hoc by
   `probe_sign.py`.
4. Every arm trains: 8-epoch smoke run, all six arms. **PASS** — epoch-5 loss
   Signed 0.263, Vanilla 0.274, CARoPE 0.581, Pos 0.542, Abs 0.733, RoPE 1.774.

---

# AMENDMENT, 2026-09-06, written while the batch was at 48/72 and before any result was read

An audit of the theory note against the local corpus found that **the central
motivating claim above — "A5 is the one axis nobody varies deliberately" — is
false.** It is not a small correction. There is a published literature on exactly
this axis, with a theorem, and it reaches into this very slot. Recording it here
rather than in the results file, so it cannot be mistaken for a post-hoc reframing.

## What the literature already has

**Sarrof et al., *The Expressive Capacity of State Space Models* (arXiv:2405.17394,
May 2024)** makes the observation I thought was mine, two years earlier and about
SSM gates rather than positional encodings:

> "Across all non-time-invariant SSMs surveyed, we find that the gate is always
> **nonnegative** ... A(x_t) ≥ 0 (nonnegative) **due to exponential or sigmoid
> parameterizations of the gate**"

That is word for word the observation in the table above — non-negativity as a
side-effect of a squashing function, never argued for — and it comes with
Theorem 2: such models cannot recognise PARITY.

**Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative
Eigenvalues* (arXiv:2411.12537, ICLR 2025)** turns it into an architectural fix:
they **prove** finite-precision LRNNs whose state-transition matrices have only
positive eigenvalues cannot solve parity, extend Mamba's and DeltaNet's range from
`[0,1]` to `[−1,1]`, and show it "consistently improves performance on
state-tracking tasks", stably at 1.3B.

**PaTH** adopts it deliberately — "We use `β_t ∈ (0,2)` as this allows for negative
eigenvalues in the transition matrix, which has been shown to boost the state
tracking performance" — and **RWKV-7** discusses it and restricts the range only
because of observed training instability.

**And Selective RoPE has already carried it into this exact slot.** Its §4.2 cites
Grazzi for the `[0,1] → [−1,1]` result and argues that "the input dependent
rotations allow it to model *flips* depending on the input", then demonstrates that
"Transformer with Selective RoPE is the only variant of Transformers capable of
solving the parity task with a single layer up to this sequence length."

So: the sign of a content-dependent phase increment, in softmax attention, has been
argued for and tested. I claimed nobody had done it. Somebody had.

## What survives, stated at its real size

1. **CARoPE, GRAPE-AP and CoPE are non-negative and none of them cites this
   literature** (0 hits for Grazzi/Sarrof/negative-eigenvalue in all three). The
   axis is known in the state-tracking community and is not being applied as a
   design constraint in the positional-encoding-for-language community. That is
   "some papers have not read others", not "nobody varies this".
2. **Nobody tests it on navigation.** The whole existing literature is parity and
   formal-language state tracking (`S_2`, `A_3`, Dyck). Parity is `cumsum mod 2`;
   net displacement on the torus is `cumsum mod N` in two dimensions. Related, not
   identical — and this project has already measured the parity leg separately
   (`ALGORITHMIC_RESULTS.md`, path integration 8/8 seeds at every length).
3. **The isolation is cleaner than what exists.** Selective RoPE's evidence is an
   architecture comparison against NoPE and RoPE; `Abs_r4` vs `Signed_r4` differ by
   one call to `.abs()` at identical parameter count.

## What this does to the verdicts

**The verdicts above are unchanged, but their INFORMATION VALUE is inverted.**

- H1 firing is now the **expected** outcome — a near-corollary of Grazzi's Theorem 2
  plus Selective RoPE's parity demonstration. It should be reported as a
  **replication in a new regime**, not as a discovery, and the note's "nobody varies
  A5" line must be deleted rather than softened.
- H1 **failing** is now the interesting result: it would mean the torus task does
  not need the sign even though parity provably does, which would separate
  "cognitive map" from "state tracking" as computational demands and would say the
  net-displacement argument is wrong about this task.

That asymmetry is the reason to let the batch finish rather than kill it. But the
headline it could ever have supported is gone, and was gone before the first
checkpoint was read.
