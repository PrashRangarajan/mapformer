# Representability vs learnability of non-negative rotary steps: prior art (2026-10-14)

Advisor plan item 5 (`docs/plans/2026-10-14_next_steps.md`, Sec. A points 1, 2, 5). Literature only, no GPU, no existing
file edited. Each claim is marked **[read]** (read first-hand in the full text, with file:line) or **[abstract]**
(abstract or search summary only). Downloads, each in its own directory (untrusted data; text from `pdftotext`):

| id | paper | local text |
|---|---|---|
| 2210.10749 | Liu et al. 2023, Transformers learn shortcuts to automata | `papers/new_2026-10-14/2210.10749/liu2023.txt` |
| 2303.06349 | Orvieto et al. 2023, LRU | `papers/new_2026-10-14/2303.06349/paper.txt` |
| 2002.09277 | Woodworth et al. 2020, Kernel and rich regimes | `papers/new_2026-10-14/2002.09277/paper.txt` |
| 2002.10487 | Amid & Warmuth 2020, Reparameterizing mirror descent as gradient descent | `papers/new_2026-10-14/2002.10487/paper.txt` |
| 2402.09963 | Hahn & Rofin 2024, Why are sensitive functions hard for transformers? | `papers/new_2026-10-14/2402.09963/paper.txt` |

Corpus files used: `papers/txt/{mamba3,grazzi,sarrof,srope,grape,mapformer}.txt`.

---

## 1. Is "a non-negative rotation can represent an inverse by wrapping" published?

**Yes, by construction, in two papers. Neither states it as a point about sign.**

- **Mamba-3, Sec. 3.2 [read]** (`mamba3.txt:576-579`): "consider the parity function on binary inputs {0, 1}, defined as
  sum_t x_t mod 2. This task can be performed using update: h_t = R(pi x_t) h_{t-1}, where R(.) is a 2-D rotation matrix.
  Such rotational dynamics cannot be expressed with real eigenvalues." The angle pi x_t is non-negative, and the element
  is its own inverse (pi = -pi mod 2 pi). There is one stepping token, so wrapping is not needed for an inverse pair.
  Mamba-3's own angle is Delta_t theta_t[i] with theta_t signed (`mamba3.txt:637`). It has no sign ablation; the
  parity contrast is data-dependent RoPE vs standard RoPE (Table 5b, `mamba3.txt:1432-1433`).
- **Grazzi et al. 2025, Thm. 3 and App. C.3 [read]** (`grazzi.txt:485-494`, `2272-2279`): "if T(A) is isomorphic to the
  cyclic group Z_m, then we can set A : Sigma -> M_2^2([-1, 1])", and the proof sets "A(w) = R(theta_w), theta_w = 2 pi
  w / m" for w in {0, ..., m-1}. **Every angle is in [0, 2 pi): the inverse of w is the non-negative angle of m - w.**
  This is the cyclic-group fact -1 = m - 1 used as a construction. App. C.1 (`grazzi.txt:2122-2130`): "Counting modulo m
  can be achieved by rotating a vector in R^2 by an angle of 2 pi / m". App. D (`grazzi.txt:2439-2441`): with generic
  rotations "an LRNN can perform addition modulo m using just one layer".
- **Grazzi's negative theorems do not apply to rotations [read].** Thm. 1 (`grazzi.txt:316-318`) requires "at least one
  eigenvalue lambda not in {x in R : x >= 0}"; Thm. 2 (`330-333`) requires "at least one eigenvalue lambda with nonzero
  imaginary part". A rotation by a non-negative angle has eigenvalues e^{+-i theta}, so it satisfies both. "Non-negative"
  in that literature means a non-negative *real eigenvalue*, not a non-negative *angle*. Sarrof et al. likewise
  (`sarrof.txt:147-148`): "the gate is always nonnegative ... A(x_t) >= 0 ... due to exponential or sigmoid
  parameterizations of the gate"; Thm. 2 (`sarrof.txt:314`) "No SSM satisfying nonnegative can recognize PARITY".
- **Liu et al. 2023 [read]**: their cyclic-group tasks use the generator set "the full set of group elements
  {0, ..., n-1}" (`liu2023.txt:1559-1561`), i.e. non-negative increments mod n with inverses by wrapping, simulated by
  attention prefix sums plus an MLP modulus (Lemma 6, `2367-2410`).
- **Selective RoPE [read]** (`srope.txt:879-939`): signed content rotations solve S2 (parity) in one layer, read as
  modelling "flips". Nothing on non-negative angles. Its implementation gates the angle with a sigmoid
  (`srope.txt:791-806`); the sign stays in W_omega q.
- **GRAPE [read]** (`grape.txt:659, 683`): its content-dependent non-negative phase is an additive logit bias (real slot),
  where non-negativity is required. It does not address rotary wrapping.
- **MapFormer [read]** (`mapformer.txt:1701-1703`): "Since the sigmoid gate can only take positive values, the model cannot
  represent an action and its inverse." This is about CoPE's counts read as interpolated positions, not a wrapping
  rotation (see `LIT_CLOCK_MAP.md`, correction 2026-10-12).
- **No source I read states the general remark** that in the rotary slot sign carries no representational content per
  channel. The per-channel version is the classical conjugate-pair fact (LRU App., `2303.06349/paper.txt:1858-1925`:
  real matrices have eigenvalues in pairs lambda, lambda*, so for a fixed rotation which member is called "positive" is a
  convention). Nothing I read extends it to content-dependent steps, where the reflection flips every token's step in
  the channel together.

**What the paper may claim as new.**
1. *May not claim:* that a non-negative rotary step can represent an inverse or a cyclic group. Cite Grazzi Thm. 3 / App.
   C.3 (theta_w = 2 pi w / m) and Mamba-3 Sec. 3.2 (R(pi x_t)), and say "the cyclic-group fact -1 = m - 1".
2. *May claim, scoped:* Prop. 1 with the rank condition (exact when at most r tokens step under a rank-r step map). It
   is a small extension: Grazzi's construction is a free per-letter table with no rank bottleneck. Present it as a
   lemma, not a contribution.
3. *May claim, as the contribution:* (a) **learnability**: in a slot where Grazzi / Sarrof's representational theorems
   do not apply, the same advice (allow sign) still holds, for an optimisation reason (0-3/8 vs 8/8 within stated
   budgets); (b) **the mechanism** (start x constraint 2x2; past-pi rescue; relative-sign statement T-1); (c) **the
   regime** (navigation maps, matched length, in-context maps with redrawn observations).
4. *Must fix:* the draft's "our sign cost replicates theirs in a navigation regime" (Sec. 5). Their cost is
   representational (real eigenvalues). Ours sits in a slot where it is not. The right phrase is "the same advice, for
   a different reason". CLAUDE.md's citable-results row ("replicates Sarrof / Grazzi / SRoPE in a new regime") has the
   same issue; flagging it, not editing it.

---

## 2. Liu, Ash, Goel, Krishnamurthy, Zhang 2023, first-hand

**What "gridworld" is [read].** Example 3 (`liu2023.txt:983-1005`): Q = {0, ..., S}, Sigma = {L, bottom, R};
delta(q, L) = max(q - 1, 0), delta(q, R) = min(q + 1, S) (`3155-3160`): "the decrement input has no effect at state 0
and the increment input has no effect at state S". **A bounded segment with walls, not a torus.** 2D is the direct
product of two bounded lines (Example 5, `1069-1073`, "rectangular gridworld ... subject to boundary effects"). Its
semigroup is not a group: "the gridworld semigroups are aperiodic" (`1434`). Theorem 3 detects the most recent wall (a
reset) with O(S) heads in 2 layers (`3163-3200`). **Theorem 3 is the wrong theorem for our torus.**

**What covers the torus [read].** The torus walk is the abelian group Z_64 x Z_64, the cyclic case:
- Lemma 6 (`2367-2410`): "there is a depth-1 Transformer which continuously simulates [the mod-n counter], with
  embedding dimension 3, width 4nT, and infinity-weight norms at most 4nT + 2". Attention computes the prefix sum. To get
  1/T rather than 1/t normalisation it uses a padding token and an absolute position embedding gamma_t = log(2T - t)
  ("It is possible to undo this normalization using the MLP"). Then "the MLP simply needs to memorize the function
  s -> (T s mod n)".
- Direct products run in parallel (Lemma 8): "Combined with Lemma 6, this immediately allows us to simulate arbitrary
  abelian groups with depth 1" (`2578-2580`).
- **Assumptions:** O(log T)-bit precision, "magnitude up to O(poly(T)), with O(1/poly(T)) approximation error"
  (`1353-1362`). Weight norms grow with T. MLP width O(nT) for the modulus, removable only "if we can add auxiliary MLP
  neurons with periodic activation functions such as x -> sin(x)" (footnote 4, `liu2023.txt:344-345`). Grazzi's summary
  of Liu (`grazzi.txt:526-527`): transformers "require either the number of layers or the precision growing with the
  input sequence length".
- **Target:** full state supervision, i.e. predict q_t at every position (`1540-1548`). Partially observed variants
  (Sec. 5.1; `1803-1806`, "Grid9 ... whether the current state is ... a boundary") are still functions of the state.
  **No construction or experiment retrieves an observation stored at an earlier visit to the current cell.** That needs
  one more attention layer that matches position codes, a composition Liu does not give (the project's own version is
  `docs/theory/2026-10-04/03_formal.md` Thm. 5.2, unverified at our width).

**What the experiments find about LEARNING [read].**
- Setup: GPT-2-like, **sinusoidal absolute PE** (`1610`), T = 100, online AdamW, depth 1-16, 20 replicates; main
  figures report the **best of 20 runs** (footnote 8, `liu2023.txt:410-412`).
- Gridworld learns easily: Grid9 best 92.2 / median 91.4 at 1 layer, 100 / 100 at 2 layers (`1640-1644`, `1700-1706`).
  It is the bounded, aperiodic case.
- Cyclic groups learn unreliably even at T = 100 with full supervision. Median of 20 (Fig. 9, `1709-1745`): C4 stays
  between 56 and 86% at 1-15 layers (99.3 at 16); C2 x C2 x C2 at 44-84% at every depth. C5, C7, C8 are >= 99% from
  3 layers. "Training is highly unstable" (Fig. 3, `381`). "high variance in training curves ... and negative progress"
  (`1647-1650`).
- "Learning from incomplete state sequences: RNNs are better" (`514-520`): with sparse labels transformers "may be unable
  to find good solutions"; LSTM is robust. Our torus loss is sparse (revisited observation positions only) and indirect.
- OOD and length generalisation fail. The parity shortcut counts and takes a modulus, and fails on counts rare in
  training (`520-550`, `1910-1935`).
- Their framing is ours: "The results in Section 3 only provide a precise understanding of representability ... To
  understand whether Transformers can actually learn these shortcuts from data ..." (`391-393`).

**Judgement of draft Sec. 4.1.** The result (1.000 vs 0.719 / 0.860 / 0.864, n = 4, complete separations) stands as
measured. Five errors in the framing:
1. "constant-depth constructions for gridworld-type automata (Liu et al., via prefix sums)" cites Theorem 3, which is
   the bounded wall-reset automaton. The torus is covered by Lemma 6 + Lemma 8 (cyclic / abelian, depth 1 for state
   simulation).
2. "So this is learnability, not capacity" is too strong. Liu's constructions do **not** establish that 4 index layers
   **at our size** can represent the task. They need MLP width ~4nT (~10^5-10^6 here; ours is 512) or periodic
   activations, an absolute position embedding (our index models have none: RoPE only, `model_baseline_rope.py`), and
   weights growing with T. They also do not include the retrieval step. Correct reading: "capacity at our width is
   not settled either way; the factorial measures what training finds within 900 epochs".
3. "The precision such constructions need grows with the walk length, which is consistent with depth helping short-gap
   revisits only" goes further than Liu supports. Liu gives O(log T) bits and poly(T) norms. The link to short-gap
   revisits is our account (THEORY_REVIEW Sec. 5: the prefix-sum route would be all-or-nothing, so depth bought a local
   route instead). Label it as an account or drop it.
4. "read at abstract level -- to verify" must go. It is now read.
5. Useful support from Liu's own data: shortcut learning for cyclic groups is unstable even at T = 100 with dense
   supervision (medians), and worse with sparse labels. The factorial is consistent with that, at 10x the length, a
   64-state modulus per axis and sparse, indirect supervision.

**Corrected Sec. 4.1 (replacement text).**

> ### 4.1 Where a map is needed, one path layer beats index attention at the depths we tried
> On the torus at T = 1024, one layer of signed path integration scores 1.000 (4/4 SOLVED) against index RoPE at one
> layer 0.719, four layers 0.860 and one block looped four times 0.864 (0/4 each, all stalled within 900 epochs; complete
> separations, p 0.029; floors 0.507 / 0.576). Depth and looping lift the index model only on revisits within 128 moves
> (0.74-0.79 -> 0.90-0.96). Long-gap and wrap-only revisits stay at 0.37-0.53 at every depth (Fig. 1). Told in English
> words: path 0.969 vs index 0.505 (p 0.0002, 7/8 vs 0/8). **This is not a capacity result. The torus walk is the abelian
> group Z_64 x Z_64, and Liu et al. (2023) show that one attention layer with an MLP can simulate such a group (a prefix
> sum followed by a learned modulus; their Lemma 6 with Lemma 8); a second layer could match the decoded position to
> retrieve the observation.** Their construction uses an absolute position embedding, an MLP whose width grows with
> n T (or periodic activations) and weights growing with T, so it does not show that our 128-wide RoPE models can
> represent the task, and we do not claim they cannot. What we measure is what training finds. Here Liu et al.'s own
> experiments point the same way: shortcuts for cyclic groups are learned unreliably even at T = 100 with every state
> labelled (median over 20 runs below 90% for C_4 at 1-15 layers), and less reliably with sparse labels. Our labels are
> sparse (revisits only) and indirect (observations redrawn per walk). On Dyck-2 at matched depth, one path layer is
> worth about three index layers (appendix F).

---

## 3. Learnability of sign-constrained parameterisations; the reflection gauge

**Zero is a wall for sign-constrained reparameterisations: textbook, cite it.**
- **Woodworth et al. 2020 [read]** (`2002.09277/paper.txt:166-174, 950-955`): for f = sum u_i v_i x_i with |u_i| = |v_i|
  at init, "their magnitudes will remain equal and their signs will not flip throughout training"; "the signs of the
  weights cannot change, since |u_i(t)| = |v_i(t)| = 0 implies u_i' = v_i' = 0." They use the "unbiased" w+^2 - w-^2
  model so that an init at zero is not "a saddle point from which gradient flow will never escape". **The draft's
  Prop. 4 (sticky wall) is this fact for |.|. Cite it; do not present it as new.**
- **Amid & Warmuth 2020 [read]** (`2002.10487/paper.txt:280-286, 467-503`): continuous GD on u with w = q_tau(u) ~
  |u|^{2/(2-tau)} equals continuous tempered-EGU on w. At tau = 0 (w = |u|) it is "the GD update (on the non-negative
  orthant)"; at tau = 1 (w = u^2/4) it is EGU, which keeps w >= 0. **Caveat:** these are per-coordinate
  reparameterisations of a weight, whereas our |.| acts on a step that is a linear function of shared latents
  (|W_out W_in x|). The analogy is per channel, not exact.
- **Dale's-law networks [abstract]** (Cornford et al., ICLR 2021; Li et al., NeurIPS 2023): sign-constrained weights are
  known to impair learning. Li et al. argue the cause is the spectrum at initialisation, not the sign constraint
  itself. This is a "start vs constraint" question like our 2x2 in another setting. Read before citing.

**Phase initialisation decides learnability for complex/rotary phases: published, opposite direction.**
- **LRU, Orvieto et al. 2023 [read]** (`2303.06349/paper.txt:71, 752-783`): eigenvalue phase parameterised
  **non-negative**, theta = exp(theta_log). Uniform theta in [0, 2 pi] on PathX: "the training loss after a few
  iterations converges to a highly suboptimal minimizer ... The model cannot recover from this suboptimal
  initialization". Fix: "restrict the range of theta to a thin slice around 0", [0, pi/10] -> 94.2%. App. E
  (`1835-1852`): exponential parameterisation helps when "the system has to learn how to 'turn'". Two relevant facts:
  (i) a strictly positive phase costs LRU nothing, because its phases are input-independent and the sign of each
  channel is a conjugate-pair convention. This is our corollary T-1(a) (one driver per channel: |.| is free), in the
  time-invariant case. (ii) A published example of phase initialisation deciding learnability: there, small phases help
  long-range tasks. Our design implication (start a non-negative step past the wrap) points the other way for a
  different reason (inverse pairs). Cite both so the reader sees the contrast.

**Learnability of sensitive / modular functions in transformers.**
- **Hahn & Rofin 2024 [read, abstract and Sec. 1]** (`2402.09963/paper.txt:12-40, 80-110`): "while transformers can
  express highly sensitive functions, such transformers are isolated in parameter space, and minima interpolating a
  sensitive function are very sharp". The torus position is a sensitive function of every move. This is a candidate
  account of why index depth does not learn it (Sec. 4.1). Only an account; not tested here.

**The relative-sign gauge (T-1) and "path through zero": not found.**
- Searched (web: RoPE sign / reflection / gauge; content-dependent non-negative phase learnability; plus the corpus).
  Nothing read states either (i) that per-channel reflection J = diag(1, -1), with J R(phi) J = R(-phi), makes the sign
  of a content-dependent rotary channel a gauge, so that |.| constrains only the *relative* signs of the tokens that
  step in it, or (ii) that the cost of non-negativity in the rotary slot is the loss of the path through zero for
  inverse pairs. The nearest published facts are the conjugate-pair convention (LRU App.; real Jordan form) and the
  sign-preservation results above.
- **Positioning:** "|s| cannot cross zero" is known (Woodworth; Amid & Warmuth). "Phase init matters" is known (LRU).
  "Non-negative rotations can represent Z_m" is known (Grazzi). What is not in what I read, and is the paper's: in
  the rotary slot the constraint removes no solution (cut circle, Prop. 3), only relative signs (T-1). Its whole cost is
  that the through-zero cancelling codes for inverse pairs lie in the excluded orthant, leaving the wrap as the only
  reachable solution, and a start past pi restores it (2x2). Claim novelty for this conjunction and the measurements,
  not for any single ingredient. Search coverage is one session of web search plus a 40-paper corpus: say "to our
  knowledge".

---

## 4. Corrected Related Work (replacement for draft Sec. 5, ~0.5 page)

> **Integrators and position.** Howard et al. (2014) drive a bank of leaky integrators by a rate alpha(t): constant,
> speed or signed velocity give elapsed time, path length or position. Their work and the Howard and Tiganj groups'
> machine-learning models keep the decay real and the rate fixed or given. GridPE (Li et al.) derives RoPE from a
> velocity-controlled oscillator, i.e. with a given velocity. MapFormer (Rambaud et al.) and TEM introduced learned
> path-integrated position for cognitive maps.
>
> **Content-dependent rotary position.** Signed content rotations: Selective RoPE (Movahedi et al.), Mamba-3 (Lahoti et
> al.; the complex-SSM / data-dependent-RoPE equivalence is their Prop. 3), and MapFormer. Non-negative: CARoPE (a rate
> in (0, 1)) and, as a gated count read through a table rather than a rotation, CoPE (Golovneva et al.). GRAPE (Zhang et
> al.) unifies the multiplicative and additive slots; its content-dependent non-negative phase is an additive bias, in
> the real (decay) slot, as in FoX.
>
> **Sign and state tracking.** Sarrof et al. (2024) and Grazzi et al. (2025) prove that recurrences whose transition
> eigenvalues are real and non-negative cannot track parity or count modulo m, and recommend allowing negative
> eigenvalues (Khavari et al. 2025 add that the sign must be input-dependent in the same layer). These theorems concern
> real eigenvalues. A rotation by a non-negative angle has complex eigenvalues and is not covered: Grazzi et al.
> implement Z_m with rotations by 2 pi w / m >= 0, and Mamba-3 writes parity as R(pi x_t) with x_t in {0, 1}. In the
> rotary slot, then, sign carries no representational content per channel (-1 = m - 1). Our question is whether it
> matters for learning. It does, and we identify why.
>
> **Sign-constrained parameterisations and initialisation.** That a reparameterised weight cannot change sign under
> gradient flow is standard (Woodworth et al. 2020; Amid and Warmuth 2020, where w = |u| is gradient descent on the
> non-negative orthant). Phase initialisation is known to decide learnability in complex diagonal recurrences: the LRU
> (Orvieto et al. 2023) parameterises phases as positive and must initialise them near zero to learn PathX. Our setting
> differs in having inverse pairs of tokens sharing a channel. There the constraint removes only relative signs, and
> the useful start is past the wrap rather than near zero.
>
> **Depth versus recurrence.** Liu et al. (2023) show that shallow transformers can represent solvable automata, with
> abelian groups in one layer via a prefix sum and an MLP modulus, using absolute positions, widths and weights that grow
> with T, and O(log T) precision. They also find that such shortcuts are learned unstably and degrade under sparse
> supervision. Hahn and Rofin (2024) tie this to sharp minima for sensitive functions. Selective RoPE and PaTH (Yang et
> al.) solve parity or Flip-Flop in one layer with content-dependent position. Our depth factorial is a learnability
> measurement in this sense, not a capacity bound.

---

## 5. Citation entries

| key | entry | status |
|---|---|---|
| howard2014 | M. W. Howard, C. J. MacDonald, Z. Tiganj, K. H. Shankar, Q. Du, M. E. Hasselmo, H. Eichenbaum. A unified mathematical framework for coding time, space, and sequences in the hippocampal region. *J. Neurosci.* 34(13):4692-4707, 2014. | read (LIT_CLOCK_MAP) |
| rambaud2025 | V. Rambaud, S. Mascarenhas, Y. Lakretz. MapFormer. arXiv:2511.19279 (v4, 10 May 2026). | read |
| sarrof2024 | Y. Sarrof, Y. Veitsman, M. Hahn. The expressive capacity of state space models: a formal language perspective. NeurIPS 2024; arXiv:2405.17394. | read |
| grazzi2025 | R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, M. Pontil. Unlocking state-tracking in linear RNNs through negative eigenvalues. ICLR 2025; arXiv:2411.12537. | read |
| khavari2025 | Khavari et al. Parity requires unified input dependence and negative eigenvalues in SSMs. arXiv:2508.07395. | abstract (complete the author list from arXiv) |
| lahoti2026 | Lahoti et al. Mamba-3. arXiv:2603.15569. | read (complete the author list and title from arXiv) |
| movahedi2026 | Movahedi et al. Selective RoPE. ICLR 2026; arXiv:2511.17388. | read |
| zhang2026grape | Zhang et al. GRAPE. ICLR 2026; arXiv:2512.07805. | read (partial) |
| veisi2025 | Veisi et al. CARoPE (context-aware rotary position embedding). arXiv:2507.23083. | read |
| golovneva2024 | O. Golovneva, T. Wang, J. Weston, S. Sukhbaatar. Contextual position encoding: learning to count what's important. arXiv:2405.18719. | read |
| yang2025path | Yang et al. PaTH attention: position encoding via accumulating Householder transformations. arXiv:2505.16381. | read |
| liu2023 | B. Liu, J. T. Ash, S. Goel, A. Krishnamurthy, C. Zhang. Transformers learn shortcuts to automata. ICLR 2023; arXiv:2210.10749. | **read (this note)** |
| hahn2024 | M. Hahn, M. Rofin. Why are sensitive functions hard for transformers? ACL 2024; arXiv:2402.09963. | read (Sec. 1); venue from memory, verify |
| orvieto2023 | A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, S. De. Resurrecting recurrent neural networks for long sequences. ICML 2023; arXiv:2303.06349. | read (Sec. 3.4, App. E); author list from memory, verify |
| woodworth2020 | B. Woodworth, S. Gunasekar, J. D. Lee, E. Moroshko, P. Savarese, I. Golan, D. Soudry, N. Srebro. Kernel and rich regimes in overparametrized models. COLT 2020; arXiv:2002.09277. | read (Sec. 1, App. B); last four authors from memory, verify |
| amid2020 | E. Amid, M. K. Warmuth. Reparameterizing mirror descent as gradient descent. NeurIPS 2020; arXiv:2002.10487. | read |
| cornford2021 | J. Cornford, D. Kalajdzievski, M. Leite, A. Lamarquette, D. M. Kullmann, B. Richards. Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units. ICLR 2021. | abstract; author list from memory, verify |
| gridpe | Li et al. GridPE. arXiv:2406.07049. | abstract (LIT_CLOCK_MAP) |

---

## 6. Edits implied for `DRAFT_workshop.md` (for the author; not made here)

1. Sec. 2 Prop. 1: open with "As in Grazzi et al.'s rotation construction of Z_m (theta_w = 2 pi w / m) and Mamba-3's
   R(pi x_t) parity, ...". Keep only the rank condition as ours.
2. Sec. 4.4 Prop. 4 (sticky wall): cite Woodworth et al. 2020 App. B / Amid & Warmuth 2020.
3. Sec. 4.1: replace with Sec. 2 above. Abstract: "beats index attention at one layer, four layers and a four-times-looped
   block *within 900 epochs*", and no wording that implies capacity.
4. Sec. 5: replace with Sec. 4 above. Delete "our sign cost replicates theirs" and "Depth and prefix sums: Liu et al.
   (2023), PaTH" as written.
5. Sec. 4.4 design implication: add a contrast clause with LRU's near-zero phase init (small phases help there; past
   the wrap helps here, because of inverse pairs).
