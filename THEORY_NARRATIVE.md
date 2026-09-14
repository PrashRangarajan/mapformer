# The where, the what, and what training finds: one account of the MapFormer theory line

Written 2026-09-12 against HEAD fbf4d09. This is a consolidation, not a source. Every number is
copied from the file named beside it; where a file carries a CORRECTED, AUDIT or WITHDRAWN block,
the block is what is quoted. Where this document and a source disagree, the source wins and this
document is wrong. Nothing here is new data.

Conventions. MDE = 2.8 sd / sqrt(n) on paired differences. A contrast inside its MDE is
**unmeasured**, never "null". "Detectable" means outside the MDE. `EM_P0` is MapEM with a single
shared position origin `p0`; `sep` is MapEM with the paper's separate `q0`, `k0`; WM is MapWM.
Rule numbers refer to `CLAUDE.md` (rules 27-34 there collide with a different 27-28 in
`RESULTS_INDEX.md`; the `CLAUDE.md` numbering is used throughout).

---

## 1. The question, and why every measurement here is a transfer measurement

The goal of this line is not positional encoding for its own sake. It is positional encoding as
the mechanism by which a model learns a relational *where*, kept separate from the *what*: the
TEM claim that structural knowledge `g` is environment-invariant, sensory content `x` is
environment-specific, memory stores the conjunction `g (x) x`, and that factorisation is what
buys transfer (`axes_measured.tex` Sec. "Factorisation, and what the separation buys";
`.claude-memory/project_state.md`).

MapFormer (Rambaud et al. 2025, arXiv:2511.19279) carries that factorisation into attention. The
where is an accumulated phase

    theta_t = omega (*) sum_{u<=t} W_out W_in x_u

which depends only on the token stream's transition structure, not on what was observed. The what
is the content embedding. Because cumsum is linear, all 64 phase channels are linear readouts of
one r-dimensional accumulator (`positional_review.tex` Sec. "The factorisation identity": checked
against the implementation at 1.5e-5).

**Every evaluation in this repository redraws the observation map from a held-out seed.** The
what-to-where binding at test time is therefore entirely new; only structure can carry over. So
every accuracy number below is already a measurement of transfer to a new instance of the same
structure. The anchor result (`RESULTS_INDEX.md`, the paper's task, parameters matched within 0.4%,
measured always-predict-blank floor 0.506):

| encoding | index position | path-integrated position |
|---|---|---|
| RoPE | 0.514 +/- 0.004 | 0.989 +/- 0.011 |
| PoPE | 0.509 +/- 0.004 | 1.000 +/- 0.001 |

An index code sits on the floor on a fresh map: it cannot locate itself, so it has nothing to
bind an observation to. A structural code arrives with the where already valid. Two boundaries
come with it (`RESULTS_INDEX.md`): the effect is a threshold in map extent (-0.010 / +0.015 /
+0.305 at 32 / 128 / 512 occupied cells, `VISITS_TEST.md`), and it falls from +0.438 to +0.050
under rotation-based actions, which allocentric recoding restores to +0.488.

What is *not* shown, and matters for the goal: transfer across a change of *structure* (topology,
action space) and faster *learning* (sample efficiency). Neither has been measured
(`axes_measured.tex`, "What is not shown").

The rest of this document asks a narrower question that the anchor raises. If the where is a
phase, what exactly is the object attention reads, what must it be able to express, how do the two
MapFormer variants combine it with content, and when the variants differ, is the difference in
what they can express or in what training finds?

---

## 2. The object: the position kernel

### 2.1 Two slots, and whose frame this is

Every rotary logit is a sum over frequency planes of magnitude times cosine of phase
(`positional_review.tex` eq. polar). A positional mechanism is a choice about one of those two
factors. Under linearity, translation invariance and continuity, the interval operators form a
one-parameter matrix group `A(d) = exp(dM)`, and Jordan form gives exactly: real eigenvalues
(decay, NoPE), conjugate pairs (damped RoPE, the phase slot), and defective generators (polynomial
factors; ALiBi is realisable this way).

**This is prior art and must be cited as such.** The classification argument is Puranik (Jane
Street blog, Apr 2026). GRAPE (arXiv:2512.07805, ICLR 2026) is the two-slot unification
(multiplicative rotations plus additive biases); its own Appendix D calls its multiplicative
extension "non-contextual", and its content dependence sits on the additive side with
`omega = g(x) >= 0`. Vetcha (2601.06113) states the same decomposition a third time. Mamba-3
(arXiv:2603.15569) publishes the content-dependent rotation itself: `Diag(A(t) + i theta(t))` with
both parts data-dependent, and a Prop. 3 titled "Complex SSM, Data-Dependent RoPE Equivalence"
(`papers/INDEX.md`; `.claude-memory/reference_positional_landscape.md`). The content-aware category
is also not unsurveyed: Zhang et al. (2503.17407, Mar 2025) Sec. 3.1.1 "content-aware position
embedding" lists CoPE and DAPE (`papers/INDEX.md`). The cell was opened on the additive side.

### 2.2 Where the mechanisms sit

| mechanism | slot | what drives it | sign of increment |
|---|---|---|---|
| RoPE | phase | index; the `Delta == 1` special case of MapFormer (`axes_measured.tex`) | +1, a clock |
| MapFormer (WM, EM) | phase | content, rank-r bottleneck `W_out W_in`, accumulated by cumsum | signed |
| Selective RoPE (2511.17388) | phase | query, conv, sigmoid gate, no bottleneck | same slot; posted 21 Nov 2025, MapFormer 24 Nov, neither cites the other (`CLAUDE.md`) |
| CARoPE (2507.23083) | phase | content, one scalar per head (rank 1 per head) | `1/(softplus+1)` in (0,1): monotone |
| Mamba-3 | phase and magnitude | data-dependent | signed |
| PoPE (2509.10534) | magnitude | content sets `mu = softplus >= 0`; phase is index | defined by deleting RoPE's content-phase term (`papers/INDEX.md`) |
| MapPoPE | both | PoPE magnitude plus MapFormer phase | signed |
| CoPE (2405.18719) | outside the per-token frame | `g_ij = sigma(q_i . k_j)`, position depends on the query-key pair, so no per-token `theta_t` exists | sigmoid: monotone |

(`RECENCY_RESULTS.md` cites CoPE as arXiv:2405.11582; the stored text `papers/txt/cope.txt` reads
2405.18719, which is used here.)

### 2.3 The kernel

For the position part of an attention score, `THEORY_KERNEL.md` Sec. 1 writes

    A_P[t,s] = q0^T R(theta_s - theta_t) k0 = sum_i a_i cos( omega_i (S_t - S_s) + phi_i ) =: kappa(dS)

with `S_t = sum_{u<=t} Delta_u` the accumulator, `a_i = |q0_i||k0_i|` and `phi_i` the per-block
phase. This gives three design axes (`EM_WM_STATE.md` Sec. 2, after audit):

| axis | sets | knobs |
|---|---|---|
| A. argument | what `dS` measures | signed vs monotone `Delta`; rank r; index (`Delta == 1`) |
| B. kernel | the shape of `kappa` | phases, magnitudes, frequencies |
| C. composition | how `kappa` meets content | EM: one shared `kappa` multiplying `A_X`. WM: a per-pair kernel whose amplitudes and phases content sets |

The frame is exact algebra for MapEM. For MapWM it holds only with per-pair amplitudes and phases
(Section 4). Its status is "a correct description of the function class", and, as Section 5 shows,
not an explanation of the one EM/WM deficit found, because that deficit is not in the function
class (`EM_WM_STATE.md` Sec. 5).

What survives of this section as the project's own contribution is empirical, not structural: the
**rank** of the content-to-angle map, and the **navigation regime**, in which none of the prior-art
papers evaluate (`.claude-memory/reference_positional_landscape.md`). The shared-vs-per-pair axis
of Section 4 is, in this corpus, isolated by no paper (`EM_WM_THEORY.md` Sec. 3: "That is this
corpus's answer, not the field's").

---

## 3. What the phase increment must be able to do

Axis A asks what `dS` measures. Three properties decide it: sign, rank, and, for tasks that ask
"k steps back", the ability to rewind.

### 3.1 Sign: a map or a clock

Attention sees only the interval sum. If increments are signed, that sum is **net displacement**
and two routes from A to B agree: the accumulator is a map. If increments are monotone, the sum is
**path length** and routes disagree: a clock (`.claude-memory/project_clock_vs_map.md`). A
cognitive map *needs* same-place-different-time to collide, which is what revisit prediction is.
Text needs the opposite, which is why a monotone clock is a reasonable object on language.

**The sign result is a replication, recorded as such before any checkpoint was read.** Sarrof et al.
(2405.17394) observed non-negative gates and a parity theorem; Grazzi et al. (2411.12537, ICLR
2025) proved and fixed the eigenvalue version; Selective RoPE Sec. 4.2 carried it into
content-dependent phase in attention (`SIGN_ABLATION.md` Sec. 5). What is new here is the
navigation regime and the isolation (`|D|` against `D`, one operation, identical parameter count).

Measured (`SIGN_ABLATION.md`, torus, 6 arms x 12 seeds, one batch, 204,757 parameters in every
MapFormer arm):

| contrast | T=512 loss-matched | T=1024 loss-matched |
|---|---|---|
| monotone (`Abs`) - signed | -0.215 (MDE 0.055, 12/12 worse) | -0.280 (MDE 0.061, 12/12 worse) |
| signed - index (RoPE) | +0.123 (MDE 0.075, 12/12) | +0.195 (MDE 0.107, 12/12) |
| monotone (`Abs`) - index | -0.092 (MDE 0.103), unmeasured | -0.085 (MDE 0.140), unmeasured |

At T=128 the signed arm sits at 1.000 +/- 0.000, so the accuracy discriminator could not show a
training-length deficit of any size; the deficit is in the training **loss**, worse for every
constrained arm on 12/12 seeds (`Abs` 0.1708 vs 0.0002). At T=128, where r(loss, acc) = -0.978,
loss-matching partials out the very deficit the constraint causes: there even the signed arm reads
-0.021 against the index code (MDE 0.013) and the monotone one -0.028 (MDE 0.025). Beyond T=128 the
monotone-minus-index contrast is unmeasured while signed-minus-index is detectable. So on
this task the whole measured value of a content-dependent phase over an index clock requires the
sign. The learned code shows why: the opposition score `||D(+x) + D(-x)|| / mean||D||` (0 = opposite
actions cancel, 2 = identical) is 0.106-0.130 for the signed arms and 1.849-1.981 for the monotone
ones; CARoPE's parameterisation reaches 1.981.

### 3.2 The crossover: a mechanism's value is set by the task

The same constraint on a k-back recency task (retrieve the k-th most recent content symbol with
uncounted filler interleaved; `k_max=64`; the answer sits 129.7 +/- 10.3 tokens back at k=64;
chance 0.0625; `RECENCY_RESULTS.md`, 6 arms x 8 seeds):

| task | contrast | raw | loss-matched | source |
|---|---|---|---|---|
| torus, T=1024 | `Abs_r4` - `Signed_r4` | -0.363 (MDE 0.025) | -0.280 (MDE 0.061) | `SIGN_ABLATION.md` |
| recency, T=2048 (T=1024 at ceiling) | `Abs_r4` - `Signed_r4` | -0.052 (MDE 0.105) | +0.016 (MDE 0.046) | `RECENCY_RESULTS.md` |
| recency, T=2048 | mean of 3 monotone arms - `Signed_r4` (registered primary) | -0.004 (MDE 0.055) | +0.026 (MDE 0.027) | `RECENCY_RESULTS.md` |

The torus row is the single `Abs` arm; the recency primary averages three monotone arms. The
like-for-like `Abs` row on recency is also unmeasured, so the reading does not depend on the choice.

Forcing a monotone increment costs 0.280 (loss-matched) on the map task and nothing measurable on
the recency task, raw or loss-matched. `RECENCY_RESULTS.md` calls the interaction
~ +0.28. The unconstrained arm learns a different accumulator per task: growth exponent alpha
(`range(S) ~ T^alpha`) 0.591 +/- 0.028 on the torus against 0.967 +/- 0.009 on recency, while every
constrained arm sits near 1.0 on both.

How the unconstrained arm counts on recency was settled by eval-only intervention, not correlation
(gate strength does not predict accuracy, r = -0.34) (`RECENCY_GATE_ABLATION.md`). Removing filler
increments costs -0.0003; replacing `Delta` with a magnitude-matched constant on content tokens
scores 0.783, on every token 0.189: **+0.594 at 8/8**. An `equalize` condition that looked decisive
(0.086) was confounded, because a control changing only theta's scale collapses as hard (0.110). A
fixed index code cannot count contextually at all: +0.750 (MDE 0.030, 8/8) for path integration
over index, a reproduction of CoPE's published claim rather than a new one.

**Scope, corrected by the audit.** The theorem written to explain this (Thm 1, "no single
accumulator is both a map and a clock") holds only for a scalar or fully constrained accumulator;
the model's is rank-r per head, so one subspace can cancel while another counts. And recency does
not *need* a clock (Section 3.4). The recency half of the crossover shows a monotone increment is
*harmless* there, not *required* (`AUDIT_2026-09-10.md` #2).

### 3.3 Rank: a conditioning failure, not a capacity one

MapFormer's `W_out W_in` has bottleneck r=2, justified in the paper by the 2D displacement vector.
Measured (`RANK_SWEEP.md`, torus, 8 seeds, one batch), against r=2 at T=1024: r=4 +0.085 (t 3.57,
8/8), r=8 +0.091, r=16 +0.079, r=32 +0.095; at T=512, +0.038 for r=4. It is a step at r=2, flat
above. r=4 costs 384 parameters and cuts the T=1024 seed sd from 0.064 to 0.012.

The cause is a skewed basis (`ACTION_GEOMETRY.md`): at r>2 the actions still occupy a 2-plane
(2-plane energy 1.0000 at r=4, 0.9996 at r=32), so the paper is right about what is expressible;
but at r=2 opposition is 0.4950 and |cos(N,E)| 0.7833, against 0.0922 and 0.1754 at r=4. The paper's
Fig. 4 reproduces three of four claims at r=2 (`PAPER_FIG4_REPRO.md`), and its own reported
limitation, |cos(orthogonal)| = 0.779, falls to 0.174 at r=4 with no regulariser. Its C4
(`||v_obs|| / ||v_action||`) does not reproduce: 0.57.

Sign and rank are one finding at two severities. Across four arms from two batches, alpha is
0.518 (signed r=4), 0.524 (`Vanilla_r4`), 0.619 (r=2), 0.943 (monotone), and
r(opposition, alpha) = +0.9995 (`LOCALISATION.md` P3). Because alpha is nearly collinear with
opposition, it adds economy, not a third cause; it is a statistic of a trained model, not a
parameter, so "vary alpha and watch degradation follow" is malformed
(`.claude-memory/project_clock_vs_map.md`).

(Two files give different torus alphas for the same arm name `Signed_r4`: 0.591 in
`RECENCY_RESULTS.md`, 0.518 in `LOCALISATION.md`. They come from different analyses; each is quoted
only beside its own comparison.)

The rank result does not transfer to MapPoPE: +0.019 at T=1024, 5/8 seeds, unmeasured
(`MAPPOPE_R4_RESULTS.md`), plausibly because PoPE has 64 frequency channels to MapWM's 32. Nor is
r=2's deficit a packing-geometry effect: on a D-dimensional torus r=2 posts its best score at D=5
(0.896 +/- 0.076) and its deficit does not grow with D (+0.110 / +0.153 / +0.055 at D=2/3/5)
(`mapformer_math.tex` Sec. "A prediction, and the paper's own 5D negative"; `DXR_RANK_THRESHOLD.md`).
The nearest published neighbour on rank is LieRE (2406.10322), whose generator-size sweep also
peaks in the interior (`papers/INDEX.md`).

### 3.4 The recency rewind

A signed accumulator can do k-back without counting upward. Let symbols carry `Delta = (1,0)`, the
query token `q_k` carry `Delta = (-(k-1), 0)`, and the MASK `(0,1)`. The inclusive cumsum makes the
query token rewind the count by k-1, so the retrieval offset is **zero for every query**. A
single-`p0` kernel (coherent, zero phase freedom), with the real omega schedule and random magnitudes
and projections, selects the answer on 1423/1423, 1417/1417 and 1415/1415 queries; without the
rewind, 0.0857 against a most-recent floor of 0.077 (`AUDIT_2026-09-10.md` #2). This is a
kernel-level construction with an idealised content gate; Section 5 upgrades it to the full model.

The consequence is the pivot of the whole line: **the retrieval offset is chosen by the model, not
fixed by the task.** Any theory that explains an EM deficit by "the offset varies per query" is
about representation, and the construction says representation is not the limit.

---

## 4. Two ways to share a kernel

### 4.1 What the code actually computes

`AUDIT_2026-09-10.md` #1, verified in `model.py:224-232`:

- **MapEM**: `softmax(A_X (*) A_P)`, `A_P = q0^T R(dtheta) k0`, built from two learned vectors. One
  kernel for every query-key pair; content cannot reshape it. `A_X` is a per-pair scalar gain.
- **MapWM**: `Q_t^T R(theta_s - theta_t) K_s = sum_b |Q_b||K_b| cos(omega_b dS + phi_b(q,k))` on
  content-derived Q, K. Amplitude and phase per block, **per pair**.

**MapWM is not additive.** The "EM is an AND-gate, WM is an OR-gate" framing, Thm 3's `A_X + kappa`
with `d score / d A_X = 1`, and the "sum" in `TALE_OF_TWO_ALGORITHMS.md` all described no model in
the repository on the WM side. The idea came from a 2026-05-10 summary never checked against the
code (`.claude-memory/feedback_em_vs_wm_mechanism.md`).

MapEM's Hadamard product is exactly TEM's conjunction:
`(A_X (*) A_P)_ts = (q^x_t (x) q^p_t) . (k^x_s (x) k^p_s)`, one bilinear form on the tensor product
of content and position spaces (`positional_review.tex` Sec. "Where MapEM and TEM-t sit", verified to
1.9e-6). So EM vs WM is precisely the question of how the where meets the what.

One qualification keeps EM's function class from being understated. The scalar gain carries a sign,
and `A_X (*) (-kappa) = (-A_X) (*) kappa` is a gauge: a negative gain turns peak retrieval into
trough retrieval (`EM_WM_THEORY.md` 1a; `AUDIT_2026-09-10.md` #5). Content can move EM's effective
target; it cannot give two pairs different kernel shapes.

The measured phase spread of the kernel across pairs (`EM_WM_THEORY.md` 1b, `probe_phase_spread.py`
after three bug fixes, 29,056 pairs per model):

| arm | circular sd of phase across pairs | live blocks |
|---|---|---|
| WM, trained | 1.947 (seeds 1.83-2.08) | 64/64 |
| WM, untrained | 2.722 | 64/64 |
| EM, single `p0` | 0.000 | 25/64 |
| EM, separate `q0/k0` | 0.000 | 64/64 |
| uniform null at this N | 3.267 | -- |

(The PAIRORIGIN probe run reports WM 1.966 and null 3.274 on its own batch, `PAIRORIGIN_RESULTS.md`.)

Only one reading is licensed: WM *can* reshape its kernel per pair and EM cannot. Training
*reduces* WM's spread, so this statistic tracks the parameterisation, and it does not show that WM's
advantage comes from reshaping.

### 4.2 [11]: what does and does not carry over

Whittington et al. (*Neuron* 113(2), 2025; MapFormer's reference [11]) say EM and WM *solutions*
are equivalent once trained, their learning dynamics differ, and EM learns faster "except on
N-back" (`TALE_OF_TWO_ALGORITHMS.md`). Its capacity result does not transfer, because MapEM has no
separate synaptic memory network: in [11]'s sense both MapFormers are WM models. And [11]'s N-back
is a fixed N with no filler (`tale_two_algorithms.txt:1672-1674`), so it corresponds to this repo's
*fixed*-k condition, not the varying-k task (`EM_WM_STATE.md` Sec. 1).

### 4.3 Where they tie and where they differ

**Ties.** MiniGrid allocentric r=4: EM - WM +0.0035; vocabulary sweep, trimmed, n_obs=256: +0.0000
(`TALE_OF_TWO_ALGORITHMS.md`).

**The difference.** On varying-k recency (`EM_WM_STATE.md` Sec. 3, `RECENCY_EM_RESULTS.md` read
with `AUDIT` #9; `k_max=64`, train T=1024, 300 epochs cosine, lr 1e-3, 1 layer, d=128):

| arm | acc T=1024 | loss < 0.5 reached | acc at k=64 |
|---|---|---|---|
| WM (`Vanilla_r4`) | 0.975 +/- 0.072 | 8/8 (median epoch 70.5) | 0.97 |
| EM sep (`VanillaEM_r4`) | 0.837 +/- 0.081 | 1/8 | 0.73 |
| EM_P0 (`VanillaEM_P0_r4`) | 0.600 +/- 0.126 | 0/8 | 0.37 |

**EM_P0 - WM = -0.375 (MDE 0.154, 0/8)**; -0.437 at T=2048.

**A second difference, qualified.** On the paper's task at extended length EM degrades less than WM.
Retrained at 50 epochs cosine with logs kept, floor-normalised (floors 0.799-0.802 at pe=0.8,
`PAPER_TASK_FLOORS.md`), EM_P0 - WM = +0.067 (l=512, MDE 0.168, unmeasured), **+0.186 (l=1024,
MDE 0.185, 8/8)** and **+0.287 (l=2048, MDE 0.194, 8/8)** (`PAPERTASK_RESULTS.md`). r(final loss,
accuracy) = -0.461, so unlike the recency line this is not a loss gap. But the pre-registered
convergence gate (IID >= 0.99) failed (WM 0.968, EM 0.985), so the registered verdict is **not
read**; the gate was probably mis-set, since WM scored 0.969 at 16 epochs. MapPoPE - EM stays
unmeasured at every length. The claim "when the offset is fixed a shared kernel is better" remains
demoted on two standing objections: the pattern is monotone in *length*, and other fixed-offset
tasks tie or reverse (`EM_WM_THEORY.md` 2a, updated).

---

## 5. Expressivity versus learnability: existence, holdability, search

The recency deficit invited a function-class story (Thm 3's corollary: "EM << WM when the offset
varies"). It is withdrawn. The replacement was built as a ladder, each rung measured.

### 5.1 Existence

With the rewind installed in the position pathway and frozen, content branch random at init,
single-`p0` EM scores **1.000 +/- 0.000 on 8/8 seeds at T=1024 and T=2048**, every k from 1 to 64
(`WARM_RESULTS.md` W1). Frozen - EM_P0 from scratch = +0.400 (MDE 0.125, 8/8); frozen - WM = +0.025
(MDE 0.071), unmeasured. The full one-layer model represents the task exactly (rule 29).

### 5.2 Holdability

Made trainable from the same install, the model fell to 0.642 +/- 0.266, back to from-scratch level
(`WARM_RESULTS.md` W4). That was first read as "the landscape rejects the solution", with an early-window
mechanism (a random content branch dismantles the rewind before content trains). Both readings are
withdrawn (`UNFREEZE_RESULTS.md`):

- Released at epoch 30 or 100, after content has trained, the rewind still breaks within 1-6 and
  1-19 epochs on every seed. The early window is refuted.
- Installed at 8x the weight scale, the same `Delta` held 0.941 +/- 0.063: +0.298 (MDE 0.270, 7/8),
  84% of the gap to frozen. Most of W4 was Adam eroding a rewind stored in tiny weights (rule 33).

The residual at 8x was content leaking into `Delta` through `w_in`. With `w_in`'s content columns
held at zero, the 8x trainable install scores **1.000 +/- 0.000 on 8/8 (0.991 at T=2048)**, against
0.941 with the leak open (`NOLEAK_RESULTS.md`). The paired contrast (+0.059, MDE 0.063) is at the
ceiling and unmeasured; the named readout was seed counts, 8/8 against 4/8. The latent pathway
settles near -0.86 with or without the leak (+0.007, MDE 0.309), and at -0.859 still gives perfect
accuracy, so slope values between about -0.85 and -1 are not degrees of failure. At the 1/64 install
the code is erased even with zero leakage (-0.008), so erosion and leakage are separable channels.

Scope: "holds" is one task, one config, n=8, with `w_in`'s content columns pinned, a point in EM's
weight space the unconstrained optimiser drifts away from.

### 5.3 Search: found per token, wrapped

`MAGONLY_RESULTS.md` had reported that no from-scratch EM arm learns a rewind: pooled linear rewind
slope 0.000 +/- 0.02, one of ~2,550 (head, block) pairs below -0.5, "0 of 40". **That readout is
withdrawn** (`SEARCH_RESULTS.md`). `theta` enters through cos/sin, so a block's rewind only needs to
hold modulo `2 pi / omega_i`, and a linear slope cannot see a wrapped solution (rule 34).

Read as a function instead (does the kernel select the answer?), 5 arms x 24 seeds
(`SEARCH_RESULTS.md` S1), for EM_P0 cells with k >= 8: 595 of 1368 solved, and of the failed cells
0.000 carry a rewind. Among solved cells in EM_P0, 0.427 rewind the query token to the kernel's
peak and 0.541 to its trough; the sign of `A_X` at the answer tracks the route exactly (1.000 /
0.000). That is the gauge of Section 4.1 visible inside trained models. Across all five EM arms,
93-97% of solved large-k cells go through the query token's own rewind.

Is the size of a rewind the obstacle? No (`SEARCH_RESULTS.md` S3, 8 seeds):

| arm | acc T=1024 | seeds >= 0.9 | acc T=2048 | median epochs to loss < 0.5 |
|---|---|---|---|---|
| EM_P0, fixed k=64 | 0.985 +/- 0.043 | 7/8 | 0.946 | 54 (8/8) |
| EM_P0, fixed k=16 | 0.994 +/- 0.018 | 8/8 | 0.966 | 25 (8/8) |
| WM, fixed k=64 | 0.947 +/- 0.120 | 7/8 | 0.755 | 99 (7/8) |

With one shared k, EM finds a 63-symbol rewind on 7/8 seeds, all wrapped (linear ratios +0.32 to
+1.42 against a target of -63), faster than WM. At T=2048 EM - WM is +0.191 (MDE 0.150, 7/8),
labelled exploratory; at T=1024 it is +0.038 (MDE 0.133), unmeasured. This is the condition that
matches [11]'s N-back, and there EM is the faster learner.

A k curriculum gains +0.127 (MDE 0.116, 7/8) without closing the gap; the whole gain is at k <= 32
(+0.300 and +0.301, both detectable), the tokens introduced by epoch 90. Tokens introduced at epochs
120 and 150 gain nothing.

So the obstacle is not representation (1.000 frozen), not holding (1.000 with the leak closed), and
not the size of one rewind (7/8 at k=64). **Varying-k recency asks for 64 separate wrapped rewinds,
one per query token, each trained only by the queries that name it**, and each token finds its
rewind or does not.

### 5.4 The readout lesson

The "0/40" error generalises. A readout not invariant to a symmetry of the model can report the
absence of a solution that is present. The relevant symmetries here were shifts by `2 pi / omega`
per block and the sign gauge `(A_X, kappa) -> (-A_X, -kappa)`. The fix is to measure the function
(does `A_P` select the answer?), not one representative of it (rule 34). The same line recorded six
readout failures, including one that reported a phase spread of exactly 0.000 for the arm whose whole
point was per-pair origins, because it branched on layer type rather than on where origins come
from (`PAIRORIGIN_RESULTS.md`, probe note).

---

## 6. Why search is hard

### 6.1 Availability: a good rewind exists for essentially every token (T1)

For a query token `q_k` the model chooses `z_k = W_in e(q_k)` in R^4 freely. A rewind is any `z_k`
making every live block near its maximum: up to 64 simultaneous congruences with different moduli in
4 unknowns, a simultaneous Diophantine approximation (`THEORY_SEARCH_AND_LENGTH.md` T1). The
decisive eval-only test computed, per token, the best achievable normalised kernel value `Q_k` in the
model's own rank-4 subspace against real episodes, on 8 stored EM_P0 seeds:

| cells (k >= 8) | n | available Q | achieved \|A\| | \|A\| >= 0.5 |
|---|---|---|---|---|
| solved (acc >= 0.9) | 166 | 0.995 | 0.544 | 0.590 |
| failed (acc <= 0.3) | 181 | 0.992 | 0.359 | 0.276 |

A near-perfect wrapped rewind is available for failed tokens exactly as for solved ones. **At
per-token grain, failures are search, not existence.** Even successes sit far below what is
available (0.544 against 0.995): the model finds a partial alignment and lets `A_X` gating carry
the rest; r(accuracy, |A|) = +0.355 over 431 cells. This is a descriptive measurement on 8
checkpoints, not an inferential contrast.

T1's second limb, "frequency pruning is the search strategy" (single-`p0` leaves 39 of 64 blocks
dead, so fewer congruences to satisfy), is withdrawn: r(dead-block fraction, tokens solved) = -0.546
over 8 seeds, the opposite sign. Underpowered, but pointing against the claim. The file's
alternative reading, that pruning broadens the kernel and is a symptom of failure, is untested.

### 6.2 The landscape: silent, then rugged

`SEARCH_RESULTS.md` S2, no training:

- **At init the position pathway is silent.** Its gradient is 1.7e-4 to 4.3e-4 of the content
  branch's (8/8 seeds); rms `A_P` is 6-9e-4 against rms `A_X` 0.31-0.34. The score is a product, so
  the pathway sits at a multiplicative saddle. The gradient's effect on the linear rewind slope has
  |t| < 2 on 7/8 seeds, and cos(gradient, direction raising the kernel at the answer) is
  -0.042..+0.023.
- **After training the path is rugged.** Counting peaks with prominence >= 1% of `kappa(0)` along the
  straight path from a token's current `Delta` to the exact rewind: median 2 at k=60-64 at init,
  15.5 in trained EM_P0. For every k >= 32 in trained EM_P0, the rewind still scores higher on the
  kernel than the point the token settled at.

The file states this as an account, not a test: a token must find its rewind before the kernel
sharpens and its path turns multi-modal. The two readouts are snapshots at init and at the end;
nothing has timed when barriers appear relative to when the gradient grows.

### 6.3 The currency: queries per token

If each token solves its own instance with only its own queries, success should track exposure per
token. `SPREAD_RESULTS.md` (EM_P0, k drawn from a set of size m, 300 epochs, 8 seeds): at a fixed
budget m=4 1.000, m=16 0.996, m=64 0.578; m4 - m64 = +0.422 (MDE 0.122, 8/8). Four times the budget
takes the full 64-offset task from 0.578 to 0.928 (two seeds stuck at 0.724 and 0.707). Its
exposure-matched cells both sat at exactly 1.000 and could not fire.

`SPREAD2_RESULTS.md` moved the design off the ceiling (60-epoch budget):

| arm | queries per token | primary acc | final loss |
|---|---|---|---|
| m4, 60 ep | 80,600 | 0.913 +/- 0.110 | 0.306 |
| m16, 240 ep | 80,600 | 0.957 +/- 0.098 | 0.295 |
| m16, 60 ep | 20,200 | 0.590 +/- 0.190 | 1.161 |
| m64, 60 ep | 5,000 | 0.248 +/- 0.175 | 2.502 |

Matched exposure: m16_e240 - m4_e60 = +0.045 (MDE 0.166), unmeasured. Fixed budget: m4 - m64 =
**+0.665 (MDE 0.213, 8/8)**. Hold exposure fixed and the number of offsets is not detectably
costly; hold the budget fixed and it is decisive.

Three limits. The matched contrast is unmeasured, not shown equal, and one of its arms sits at
0.957. Exposure was varied through the budget, so "queries per token" and "optimiser steps per
token" are not separated. And r(final loss, accuracy) = -0.936 (SPREAD2) and -0.945 (SPREAD): these
are fit contrasts. WM at the 4x budget was never run, so -0.375 remains the fair shared-budget
comparison.

### 6.4 What phase freedom does, as far as is known

Letting `k0`'s per-block phases move (`AlignFree`) against a control that is the same function at
init with the same optimiser scale and pinned phases (`MagOnly`): **+0.146 (MDE 0.086, 22/24)**, and
+0.113 on fresh seeds 8-23 alone (MDE 0.111, 14/16, clearing by 0.002) (`MAGONLY_RESULTS.md`).
Magnitude freedom that the optimiser does move is unmeasured (-0.015, MDE 0.083); initial coherence
is unmeasured (-0.004, MDE 0.076; `D5_RESULTS.md` via `EM_WM_STATE.md`).

What it changes (`SEARCH_RESULTS.md` S1): every free-phase head peaks off zero (48/48), but the route
does not change (peak-rewind fraction 0.445 vs 0.425) and it does not shorten the shift. Per-token
success is roughly flat at ~0.75 across distance from the kernel peak in free-phase arms, where
coherent arms fall with distance. The gain by k bin is detectable at 1-16 (+0.124, MDE 0.084) and
17-32 (+0.263, MDE 0.110), unmeasured at 33-64. **The mechanism is unidentified.** It raises
per-token success at every distance; why is not known. On recency, r(loss, acc) = -0.978 over the
MagOnly batch.

### 6.5 Length: collisions as a switch (T2)

Five mechanisms help specifically at OOD length: rank, the InEKF, the forget gate, PoPE, and EM over
WM on the paper task. T2 proposed that failure at length is a collision: some distractor key outranks
the answer on the position kernel. Tested on EM_P0 paper-task checkpoints, 4 seeds x 4 lengths
(`THEORY_SEARCH_AND_LENGTH.md` T2 results; `N_coll` = prior keys at a different cell ranked at least
as high as the correct key by the model's own kernel):

| length | collision rate | accuracy | acc given 0 collisions | acc given >= 1 |
|---|---|---|---|---|
| 256 | 0.058 | 0.974 | 0.983 | 0.825 |
| 512 | 0.085 | 0.974 | 0.989 | 0.817 |
| 1024 | 0.132 | 0.954 | 0.981 | 0.777 |
| 2048 | 0.174 | 0.933 | 0.964 | 0.786 |

Pooled: **0.973 with no collision, 0.787 with any.** Being outranked by one distractor is associated
with a 0.19 loss of accuracy, and the collision rate triples over the range. Three things fail:

- **The pigeonhole dose form.** There is no dose-response above one collision (1-2: 0.743; 129+:
  0.797). The operative event is the kernel failing to rank the correct key first, a switch.
- **"Length acts only through collisions."** At matched collision status longer is still worse, and
  the collision rate accounts for 53% of the 0.041 drop from 256 to 2048 steps.
- **The cross-arm half is untested.** Whether a weights-derived resolution orders Vanilla, EM_P0 and
  MapPoPE by breakdown length has not been run. For WM there is no position-only score on which to
  count collisions.

This is an association within one arm, not an intervention.

---

## 7. What per-pair freedom buys

If EM's recency deficit is the cost of a shared kernel (one kernel, so every token must move its own
`Delta` to reach it), then giving EM per-pair position origins should remove the need to rewind.
`EMPair_r4` sets `q^p_t = p0 + W^q_out W^q_in x_t` (likewise for k), keeping Hadamard composition,
rank and depth; `W_out` is zero-initialised, so at step 0 it is EM_P0 exactly (max |logit diff|
0.000e+00) with +2,048 parameters (+0.92%) (`PAIRORIGIN_RESULTS.md`).

At n=8 it scored 0.880 +/- 0.136 against EM_P0 0.600 and WM 0.975: +0.280 (MDE 0.216, 7/8);
EMPair - WM = -0.095 (MDE 0.135), unmeasured.

The capacity control `EMPairConst_r4` uses the same pathway reading a learned constant, so origins
are identical for every token and the kernel stays shared, with 128 *more* parameters than EMPair
(224,537 vs 224,409) (`PAIRCONST_RESULTS.md`). It landed between the two, and at n=8 neither half of
the split was detectable. Extended to n=48 (`PAIRSPLIT_RESULTS.md`; seeds 0-7 reused under a bitwise
determinism re-check, EM_P0 extended to 48):

| term | delta | MDE | seeds + | verdict | share |
|---|---|---|---|---|---|
| total, EMPair - EM_P0 | +0.215 | 0.068 | 42/48 | detectable | 100% |
| pathway, EMPairConst - EM_P0 | +0.124 | 0.071 | 34/48 | detectable | 58% |
| freedom, EMPair - EMPairConst | +0.091 | 0.066 | 34/48 | detectable | 42% |

The freedom term replicates on the fresh seeds 8-47 alone (+0.089, MDE 0.069). The total and the
pathway term were inflated at small n (total +0.280 at n=8, +0.191 at n=24; pathway +0.182 at n=8,
+0.100 at n=24). r(final loss, accuracy) = -0.954 over the 96 runs: fit contrasts.

The mechanism readouts separate the two terms cleanly (`PAIRCONST_RESULTS.md`):

| arm | phase spread across pairs | solved cells via a per-token rewind |
|---|---|---|
| EM_P0 | 0.000 | 0.964 |
| EMPairConst | 0.000 | 0.948 |
| EMPair | 1.448 | 0.189 |

(`PAIRORIGIN_RESULTS.md`'s own run of the route probe gave 0.962 and 0.185 for EM_P0 and EMPair.)

Read together, by intervention: **extra capacity in the position pathway buys about +0.12 and leaves
the mechanism untouched; making those parameters read the token buys about +0.09 more and changes
the mechanism completely**, since the model stops rewinding query tokens and shapes the kernel per
pair instead. The kernel-sharing account of the varying-offset deficit is supported, with the
freedom share smaller than the n=8 headline suggested.

What this does not say. It is evidence about the varying-offset case only; it does not revive
"a shared kernel is better when the offset is fixed" (`EM_WM_THEORY.md` 1d). It does not show that
WM wins recency *through* per-pair freedom: WM's trained spread (1.947) is below its untrained
spread (2.722), and EMPair - WM is positive on 0/8 seeds (`PAIRORIGIN_RESULTS.md`). And it
leaves open the relation to phase freedom in `q0/k0` (Section 6.4), a smaller freedom (n_b global
phases rather than per pair) whose mechanism is unidentified. The ordering WM > EM-sep > EM-P0 on
recency tracking phase freedom is a post-hoc hypothesis, untested (`AUDIT_2026-09-10.md`).

---

## 8. What is not explained

Each item names the test that would settle it. Tests quoted from a source file say so; the rest are
proposals and are labelled.

1. **Why search fails given availability.** T1 shows a good rewind exists for failed tokens; S2 shows
   a silent start and a rugged end. The window account is untested.
   *Test* (`SEARCH_RESULTS.md` (b)): record per-token rewind status and path ruggedness every epoch
   from scratch, to time barriers against gradient growth. Also (`SPREAD2_RESULTS.md` caveat): the
   same exposure at different batch sizes, to separate queries per token from optimiser steps per
   token. And (`SPREAD_RESULTS.md`): WM at the 4x budget, to say how much of -0.375 is convergence rate.

2. **What phase freedom does.** +0.146, no route change, flat per-token success across distance.
   *Test* (`EM_WM_STATE.md` Sec. 6, eval-only): per-offset `A_P` argmax and learned `kappa` shape for
   `AlignFree` vs `MagOnly` on existing checkpoints.

3. **The T2 residual** (47% of the 256 -> 2048 drop in the one arm probed). The file names no test.
   *Proposal*: the T1 file's untested link, that pruning broadens the kernel and costs resolution,
   is measurable on the same checkpoints (kernel autocorrelation width against collision-free
   accuracy at each length).

4. **The T2 cross-arm test.** *Test* (`THEORY_SEARCH_AND_LENGTH.md`): compute resolution and
   accumulator spread per checkpoint for Vanilla, EM_P0 and MapPoPE-Flat and check whether the
   predicted collision rate orders the arms by breakdown length with one global margin. A WM version
   needs a position-only score that WM does not have; that design problem is unsolved.

5. **Why OOD length is the universal signature.** Rank, the InEKF, the forget gate, PoPE and EM all
   help there. alpha covers sign and rank only; the forget gate (-0.051, MDE 0.092), PoPE (+0.006,
   MDE 0.177) and Level 1.5 (+0.009, MDE 0.191) leave it unmeasured (`ACCUMULATOR.md`). The imported
   critical-dimension account is refuted (`LOCALISATION.md`). One observation worth keeping beside
   this: the forget gate's gain exists at r=2 (+0.086 loss-matched, T=1024, 8/8) and not at r=4
   (-0.002), with a raw interaction of -0.082 (2/8; no MDE given) (`FORGET_GATE.md`). Some of the five
   may be repairing the same deficit. *Test*: item 4, extended to the rank and forget-gate arms.

6. **The forget gate as a second, monotone clock.** Its `sum log gamma` grows with alpha +0.956
   (`ACCUMULATOR.md`); it needs a live lambda (frozen: -0.016, MDE 0.118, `FORGET_CONTROL.md`) and the
   gain is anti-correlated with lambda (r = -0.516, `FORGET_GATE.md`); the transient-aid story is
   refuted (r(peak decay, gain) = -0.531, `LAMBDA_TRACE.md`). Mechanism unidentified.
   *Test* (`FORGET_CLOCK_PREREG.md`, `run_forget_clock.sh`): the gate should help more on recency at
   T=4096 than on the torus. The batch completed once and was deleted by mistake; it must be re-run.

7. **Whether EM's extended-length advantage on the paper task is real under a gate that can pass.**
   *Test* (proposal, following `PAPERTASK_RESULTS.md`): re-register the convergence gate at a level
   WM can reach and re-read P1; the length-vs-offset objection needs a fixed-offset task where length
   is held constant.

8. **`|rho|` at learned amplitude.** Thm 2's +0.292 (MDE 0.063, 8/8, torus) is for a frozen kernel at
   peak amplitude ~0.003, where learned kernels reach 0.06-0.08 (torus) and 0.27-0.50 (recency)
   (`AUDIT_2026-09-10.md` #4). *Test* (`EM_WM_THEORY.md` P7): frozen plus vs zero at learned amplitude,
   2 arms x 8.

9. **Rank below and at the threshold.** r=3 and r=1 have never been run; "r=1 -> 0.66" has no
   experiment behind it (`RANK_SWEEP.md`). *Test*: one arm each, 8 seeds.

10. **The factorisation thesis itself.** Transfer across a change of structure, and sample-efficiency
    curves (`axes_measured.tex`). Neither exists. Everything in Sections 1-7 is transfer to a new
    instance of the same structure, measured as final accuracy.

11. **The map-size threshold** (flat at 32 and 128 occupied cells, +0.305 at 512). Nothing in the
    kernel frame has a knee there (`THEORY_KERNEL.md` Sec. 7). No test is specified.

---

## 9. The graveyard

Withdrawn or refuted theoretical claims. Do not revive them.

| claim | what killed it (file, number) | lesson |
|---|---|---|
| MapWM is additive: EM is an AND-gate, WM an OR-gate; Thm 3 `d score/d A_X = 1` | `AUDIT_2026-09-10.md` #1: `model.py:224-232` rotates content Q,K; per-pair amplitudes and phases | Rule 21 (read the code, not a summary of it) |
| Thm 3 corollary: EM << WM when the offset varies per query (a representational limit) | `AUDIT` #2: a single-`p0` kernel solves recency 1423/1423 once the query token rewinds; full model frozen 1.000, 8/8 (`WARM_RESULTS.md`) | Rule 29 (existence before mechanism) |
| "EM never finds the rewind from scratch (0/40)" | `SEARCH_RESULTS.md`: solved cells rewind to peak (0.427) or trough (0.541), wrapped; fixed k=64 found 7/8 | Rule 34 (readouts must respect the model's symmetries) |
| W4: the landscape rejects the solution; a random content branch dismantles it in an early window | `UNFREEZE_RESULTS.md`: release at epoch 30 breaks within 1-6 epochs; 8x install +0.298 (MDE 0.270, 7/8), 84% of the gap | Rules 32, 33 (existence then stability; install at training scale) |
| U4: two recorded channels are exhaustive | `UNFREEZE_RESULTS.md` top block: the latent code has two coordinates; full pathway -0.866 vs coordinate 0 -0.989 | Enumerate every trainable route (unnumbered, `EM_WM_STATE.md` Sec. 7) |
| The position effect scales with observation aliasing | `ALIASING_CONTROLLED.md` design; at fixed grid 32, +0.178 (32 cells/token) vs +0.305 (2 cells/token, 800 ep, `VISITS_TEST.md`, `CLAUDE.md` 2026-08-30): sign inverted | Rules 5, 10 (budget extensions over conditioning arguments) |
| Critical-dimension import: OOD damage lives in under-trained low-frequency channels | `LOCALISATION.md` P2: ablating them costs -0.001 at T=128 and -0.139 at T=1024 (`Signed_r4`), the opposite sign | Rule 17 (check the premise applies here) |
| The InEKF's wrap bounds the accumulator | `ACCUMULATOR.md` P1: it wraps the innovation; range(theta_hat) 285.6 vs range(theta_path) 283.9 | Rule 21 |
| r=2's deficit is packing geometry (the account of v4 Table 6) | `mapformer_math.tex`, `DXR_RANK_THRESHOLD.md`: r=2 best at D=5 (0.896); deficit +0.110 / +0.153 / +0.055; and the paper set r=D in 5D | Rule 25 (check what the paper actually did) |
| Level 1.5 decomposes into named parts (wrap, per-token gate, measurement) | `L15_ABLATION.md`, n=5: no single component detectable; "ConstR worse than nothing" sign-inverted; only Level15 - Vanilla at OOD length, loss-matched (t 3.08 / 3.83) | Rule 6 (three seeds is not a point estimate; this rested on one); rule 9 |
| T2 pigeonhole dose form: accuracy falls with the number of competing keys | `THEORY_SEARCH_AND_LENGTH.md`: 1-2 collisions 0.743, 129+ 0.797 | Rule 19 (split the hypothesis) |
| Frequency pruning is the search strategy (T1) | `THEORY_SEARCH_AND_LENGTH.md`: r(dead fraction, tokens solved) = -0.546, n=8 | Rule 19 |
| Length acts only through collisions | Same file: collisions explain 53% of the drop; accuracy given 0 collisions still falls 0.983 -> 0.964 | Rule 19 |
| n=8 effect sizes as point estimates | `sep - P0` +0.237 -> +0.128 at n=24 (`AUDIT` #3); MagOnly +0.213 (seeds 0-7) -> +0.113 fresh (`MAGONLY_RESULTS.md`); magnitude term +0.120 -> -0.033 (`AUDIT` stale list); PAIRORIGIN total +0.280 -> +0.215 and pathway +0.182 -> +0.124 (`PAIRSPLIT_RESULTS.md`, "fourth instance") | Rule 27 (same-seed agreement is determinism); report fresh seeds beside pooled |
| Selective RoPE's sigmoid gate helps by suppressing observation tokens | `GATE_PROBE.md`: 1.35x on the torus where it helps, 1.54x on parity where it hurts; and the per-knob arms also delete omega (`.claude-memory/project_rank_and_selective_rope.md`) | Rule 23 (verify what a probe and an arm measure) |
| Coherence (Thm 2 corollary) is a clock/map choice that inverts by task | `N5_RESULTS.md` via `EM_WM_STATE.md`: same sign on both tasks, interaction +0.113 (MDE 0.195) | Rule 19 |
| The sign of the kernel (`rho = +1` vs `-1`) matters | `AUDIT` #5: `A_X` absorbs it; the contrast had expectation zero | Rule 28 (check for gauges) |
| D4 "reproduced +0.237 to three decimals" | `AUDIT` #10: 16/16 bitwise-identical checkpoints | Rule 27 |
| Phase freedom helps EM find the rewind | `SEARCH_RESULTS.md` S1: route unchanged (0.445 vs 0.425) | Rule 19 |
| alpha is an independent diagnostic that can be varied to test degradation | `.claude-memory/project_clock_vs_map.md`: r(opposition, alpha) = +0.9995; alpha is a fitted statistic, not a parameter | Rule 19 |
| An explicit what/where gate should help, since the learned gate is load-bearing | `GATED_RESULTS.md`: 4.16x separation, +0.004 torus (MDE 0.008), -0.039 recency (MDE 0.079) | "Load-bearing" does not imply "needs help" (unnumbered) |
| "EM wins wherever the offset is fixed" | `EM_WM_THEORY.md` 2a: monotone in length; ties or reversals on MiniGrid, vocab, Match-Query | Rule 12 (seeds on the comparison claimed) |

---

## 10. Claims ledger

| # | claim | status | source |
|---|---|---|---|
| 1 | Two-slot classification; content-dependent rotation; content-aware PE as a category | PRIOR ART (Puranik; GRAPE; Mamba-3; Zhang 2503.17407) | `papers/INDEX.md` |
| 2 | Path integration, not the encoding, carries in-context maps on a redrawn observation map | ESTABLISHED (index arms on the 0.506 floor; path-integrated 0.989 / 1.000) | `RESULTS_INDEX.md` |
| 3 | Signed increment load-bearing on navigation: signed - index +0.195 loss-matched at T=1024; monotone - index nowhere positive | ESTABLISHED in navigation (n=12, MDE 0.107, 12/12); the sign axis itself PRIOR ART (Sarrof, Grazzi, Selective RoPE) | `SIGN_ABLATION.md` |
| 4 | Crossover: monotone costs -0.280 (torus, LM, T=1024) vs -0.004 raw / +0.026 LM (recency, T=2048) | ESTABLISHED on torus (n=12, MDE 0.061); recency half UNMEASURED (MDE 0.055 / 0.027) | `SIGN_ABLATION.md`, `RECENCY_RESULTS.md` |
| 5 | Content gate is most of the recency counting mechanism (+0.594, magnitude-matched) | ESTABLISHED by intervention (n=8, 8/8; not pre-registered, control for a confound) | `RECENCY_GATE_ABLATION.md` |
| 6 | Index codes cannot count contextually (+0.750) | ESTABLISHED (n=8, MDE 0.030); reproduces CoPE, PRIOR ART | `RECENCY_RESULTS.md` |
| 7 | r=4 over r=2 on MapWM: +0.085 at T=1024 | ESTABLISHED (n=8, t 3.57, 8/8) | `RANK_SWEEP.md` |
| 8 | r=2 fails by a skewed basis (opposition 0.495 vs 0.092) | ESTABLISHED as description of learned codes (8 seeds); causal route not tested | `ACTION_GEOMETRY.md` |
| 9 | r=4 helps MapPoPE | UNMEASURED (+0.019, 5/8) | `MAPPOPE_R4_RESULTS.md` |
| 10 | alpha tracks opposition (r = +0.9995); explains sign and rank, not forget gate / PoPE / InEKF | EXPLORATORY (four arms, correlational); the three non-changes UNMEASURED (MDE 0.092-0.191) | `LOCALISATION.md`, `ACCUMULATOR.md` |
| 11 | MapWM has a per-pair kernel, MapEM a shared one; Hadamard = TEM's `g (x) x` | ESTABLISHED (code and algebra) | `AUDIT_2026-09-10.md` #1; `positional_review.tex` |
| 12 | EM_P0 - WM on varying-k recency = -0.375 | ESTABLISHED (n=8, MDE 0.154, 0/8) | `EM_WM_STATE.md`; `AUDIT` #9 |
| 13 | EM and WM tie on MiniGrid (+0.0035) and vocab (+0.0000) | DIRECTIONAL (no MDE quoted in source) | `TALE_OF_TWO_ALGORITHMS.md` |
| 14 | EM degrades less than WM with length on the paper task (+0.287 floor-normalised at l=2048) | EXPLORATORY: detectable (MDE 0.194, 8/8) but registered gate failed, verdict not read | `PAPERTASK_RESULTS.md` |
| 15 | EM represents recency exactly (frozen install 1.000) | ESTABLISHED (n=8, 8/8; +0.400 over scratch, MDE 0.125) | `WARM_RESULTS.md` |
| 16 | Trainable EM holds the rewind at 8x with the leak closed (1.000) | ESTABLISHED by seed count (8/8 vs 4/8); paired contrast UNMEASURED (+0.059, MDE 0.063) | `NOLEAK_RESULTS.md` |
| 17 | Found from scratch per query token, wrapped, peak or trough, `A_X` sign tracking the route | ESTABLISHED as description (5 arms x 24 seeds; failed cells 0.000) | `SEARCH_RESULTS.md` S1 |
| 18 | One shared k=64 rewind is findable (0.985, 7/8) and EM learns it faster than WM (54 vs 99 epochs) | ESTABLISHED for findability (n=8); EM - WM at T=2048 +0.191 EXPLORATORY (MDE 0.150); at T=1024 UNMEASURED | `SEARCH_RESULTS.md` S3 |
| 19 | Curriculum +0.127, confined to k <= 32 | ESTABLISHED (n=8, MDE 0.116, 7/8) | `SEARCH_RESULTS.md` S3 |
| 20 | Failures are search, not availability (Q 0.992 failed vs 0.995 solved) | ESTABLISHED as description (8 checkpoints, 347 cells; eval-only) | `THEORY_SEARCH_AND_LENGTH.md` T1 |
| 21 | Silent at init, rugged after training; a timing window | EXPLORATORY (two snapshots; window untested) | `SEARCH_RESULTS.md` S2 |
| 22 | Fixed budget: fewer offsets better (m4 - m64 +0.665) | ESTABLISHED (n=8, MDE 0.213, 8/8) | `SPREAD2_RESULTS.md` |
| 23 | Queries per token is the currency (matched +0.045) | DIRECTIONAL/UNMEASURED (MDE 0.166); exposure and steps not separated | `SPREAD2_RESULTS.md` |
| 24 | Phase freedom in `k0`: +0.146 vs MagOnly | ESTABLISHED (n=24, MDE 0.086, 22/24; fresh +0.113, MDE 0.111); mechanism unidentified | `MAGONLY_RESULTS.md` |
| 25 | Magnitude freedom; initial coherence | UNMEASURED (-0.015, MDE 0.083; -0.004, MDE 0.076) | `MAGONLY_RESULTS.md`; `EM_WM_STATE.md` |
| 26 | Per-pair origins: total +0.215 = pathway +0.124 + freedom +0.091 | ESTABLISHED (n=48; MDE 0.068 / 0.071 / 0.066); freedom replicates on fresh seeds (+0.089, MDE 0.069) | `PAIRSPLIT_RESULTS.md` |
| 27 | Freedom, not capacity, abandons the per-token rewind (0.948 -> 0.189; spread 0 -> 1.448) | ESTABLISHED by intervention (n=8 mechanism readout, control with more parameters) | `PAIRCONST_RESULTS.md` |
| 28 | EMPair recovers to WM | UNMEASURED (-0.095, MDE 0.135, 0/8) | `PAIRORIGIN_RESULTS.md` |
| 29 | WM wins recency because its kernel is per-pair | UNTESTED hypothesis (trained spread below untrained) | `EM_WM_THEORY.md` 1b |
| 30 | Collisions are a switch (0.973 vs 0.787) and account for 53% of the length drop | EXPLORATORY (association; one arm, 4 seeds); cross-arm half untested | `THEORY_SEARCH_AND_LENGTH.md` T2 |
| 31 | `\|rho\|` matters (+0.292) | ESTABLISHED only for a frozen kernel at ~0.003 amplitude (n=8, MDE 0.063) | `AUDIT` #4 |
| 32 | Explicit what/where gate separates (4.16x) and buys nothing | Separation ESTABLISHED (8/8); accuracy effects UNMEASURED | `GATED_RESULTS.md` |
| 33 | Forget gate +0.086 at r=2 | ESTABLISHED (n=8, 8/8 loss-matched); mechanism unidentified; clock account UNTESTED | `FORGET_GATE.md`, `FORGET_CONTROL.md`, `LAMBDA_TRACE.md` |
| 34 | Level 1.5 helps at OOD length (loss-matched t 3.08 / 3.83), no named component load-bearing | ESTABLISHED for the total at n=5 after loss-matching; components UNMEASURED | `L15_ABLATION.md` |
| 35 | Thm 3, its corollary, additive WM, 0/40, early window, aliasing, critical dimension, InEKF bound, Table 6 geometry, pruning, pigeonhole dose, collisions-only, gate-as-suppressor | WITHDRAWN | Section 9 |
| 36 | Transfer across a change of structure; sample efficiency | UNTESTED | `axes_measured.tex` |

A note that applies to every recency-line row (12, 15-28): r(final loss, accuracy) runs from -0.936
to -0.983 in these batches. Those contrasts are statements about what training finds and fits, which
is the frame the existence construction licenses; loss-matching cannot separate optimisation from
representation there, because it conditions on a mediator (`AUDIT_2026-09-10.md` #7).

---

## Summary of the argument

The where in MapFormer is an accumulated, content-driven, signed phase, and attention reads it
through a position kernel. The two-slot frame and the content-dependent rotation are published; what
this line adds is empirical. For the kernel to support a map, the increment must be signed and
well-conditioned (sign and rank, one mechanism at two severities), and its value depends on the task
(the crossover). MapWM gives each query-key pair its own kernel; MapEM shares one. They differ
detectably on one varying-offset retrieval task, and there the difference is not expressivity. The
solution exists, can be held, and is available to every query token, but each token must find its
own wrapped rewind with only its own queries, from a start where the position gradient is silent to
an end where the path is rugged. Per-pair origins remove the need to search that way and recover
most of the gap, split between extra pathway capacity and genuine per-pair freedom. What remains
unexplained is why search fails when solutions are available, what phase freedom does, and why so
many unrelated mechanisms help specifically at length; in the one arm probed, collisions account for
about half of its own length drop. The factorisation thesis the line was built to serve has been measured only
as transfer to a new instance of the same structure.
