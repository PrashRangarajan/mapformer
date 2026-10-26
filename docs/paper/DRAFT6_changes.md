# Draft 6 vs draft 5: changes, sources, word counts (2026-10-16)

Draft 6: `docs/paper/DRAFT6_workshop.md` (main text + references + figure list), `docs/paper/APPENDIX_draft6.md` (A-D, F-K),
`docs/paper/APPENDIX_theory_v3.md` (E). Draft 5 and its appendices are untouched. One results file edited: a two-line
CORRECTED banner at the top of `GAUGE_RESULTS.md` (body unchanged). Inputs: the five-agent review of draft 5
(`docs/paper/review_draft5/`: `SYNTHESIS.md` section B items 2-10, `MOCK_REVIEW_draft5.md`, `NUMBERS_AUDIT.md`,
`THEORY_CHECK.md`, `PRIOR_ART.md`, `REPRO_AUDIT.md`), `OUTLINE_overall.md`, and the paper texts under `papers/` (each new
citation checked first-hand, below). No GPU, no training; one new number source (`docs/audits/2026-10-16/paper2x2_T128_floors_out.txt`,
committed before this draft) and the referee's `theory_check_extra.py` (P1(f), A0). Nothing committed.

Source keys: **TC** = THEORY_CHECK, **NA** = NUMBERS_AUDIT (error number), **PA** = PRIOR_ART, **RA** = REPRO_AUDIT (note
number), **MR** = MOCK_REVIEW, **SY** = SYNTHESIS item, **T** = the task brief.

## Word counts

`wc -w` from "## Abstract" to the line before "## References" (the draft-5 method; headings, tables and formula tokens
included).

| section | draft 5 | draft 6 |
|---|---|---|
| Abstract | 211 | 225 |
| 1. Introduction | 338 | 361 |
| 2. Setup and terms | 372 | 332 |
| 3. Theory | 618 | 517 |
| 4. Results (heading) | 3 | 3 |
| 4.1 | 440 | 452 |
| 4.2 | 210 | 190 |
| 4.3 | 419 | 377 |
| 4.4 | 103 | 67 |
| 4.5 | 289 | 250 |
| 4.6 | 197 | 180 |
| 5. Related work | 279 | 352 |
| 6. Limitations | 250 | 191 |
| **main text** | **3,729** | **3,497** |

Margin: 3 words. The placeholder "[NUMBERS FROM FIGS AGENT]" (4 words, Sec. 4.1) must be replaced by the usage-weighted
shares at about the same length, or the sentence's unweighted numbers dropped.

Moved out of the main text to make room (destination): the "Three integrators" paragraph of Sec. 3 (appendix E, end of A0,
with TC's per-channel Phi fix; one sentence stays in Related work); the "Depth" paragraph of Sec. 5 (appendix K, end); the
untested torus account of Sec. 4.4 and its "ring result, not a design rule" sentence (appendix C8 caption); details of
"What fits loosely" (now "Loose ends"; the 24-42% collapse share and the toy's |.| side stay in C12 / I); "(a periodic
position table read at a gated count)" and CoPE's +0.157 over the index (appendix H); "a third observation-step-free ring
batch" (A5); "Index RoPE uses the canonical ladder" (A2). Nothing with a number was dropped from the paper as a whole.

## 1. THEORY_CHECK fixes (TC), 6 GAP/WRONG + 16 wording items, all applied

| # | location | old | new | source |
|---|---|---|---|---|
| 1 | E A0 | "F(t) = ... sum_u e^{-s (Phi_t - Phi_u)} Delta_u f_u, exactly, for piecewise-constant f" | weight ((1 - e^{-s Delta_u})/s) f_u, exact under zero-order hold; Delta_u is the first-order term, exact for impulse input; check line added (RK4 vs corrected 1.1e-14, vs old 0.857) | TC 1, WRONG |
| 2 | main Sec. 3 P4 | "with a = omega \|s\|, dL/ds = sign(s) dL/da, so ... s = 0 is a strict local minimum" | "dL/ds = omega sign(s) dL/da ... a strict local minimum along s, other parameters fixed (cf. Amid and Warmuth 2020, Prop. 2)" | TC 7, WRONG |
| 3 | E P4 (b), (c), check line | "strict local minimum of L"; "(c) Gradient descent with step eta ..."; "dL/ds = sign(s) dL/da exactly" | "along s, with every other parameter fixed" + why it is not a loss minimum (shared s = w_c . z_x); "(with l'(0) != 0) where the signed loss decreases into s < 0"; (c) needs mu > 0, \|s_0\| <= rho, GD on s (factor-GD remark added); check reads "dL/d\|s\|" | TC 7 |
| 4 | main Lemma 1(ii) | "the barrier excludes no map code where one with sign-coherent rows exists" | "It excludes no map *function* that has a sign-coherent code, which needs no wrapping" | TC 3, WRONG |
| 5 | E T1(ii) extension | "The same holds whenever the task has a solution code whose rows are sign-coherent" | "for map *functions*", with the +1/+2 counterexample code (1, 2 - 2 pi) excluded as a parameter point | TC 3 |
| 6 | app. H (CoPE) | "move-gate difference an odd multiple of P/32" | "alpha an odd multiple of P/32, i.e. move-gate difference an odd multiple of P/16" | TC 11, WRONG |
| 7 | E P7 precision | "(proved as a sufficient condition)"; "tau sqrt(N) < P/64" | first two terms proved sufficient; observation term "N max_o \|dg_o\| < P/64 is sufficient; tau sqrt(N) is the typical size (a heuristic, not a bound)"; summary-table status updated | TC 11, GAP |
| 8 | softplus (main Sec. 3, abstract, contribution 2, Sec. 6; D4; E new P1(f); summary table) | "whether the softplus arms can represent the map ... was not tested"; D4 "its representability is untested"; Sec. 6 "untested representability" | exact construction P1(f): observations annihilated, omega' = 2 pi / ln 2, W_out' = softplus^{-1}((remainder + 2 pi)/omega'); built in `MapFormerWM_Pos_r4` from SO_s800, logits to 2.8e-11, angle error 8.4e-15; "existence under training was not tested" everywhere | TC 4/item 5, GAP; T item 2 |
| 9 | main Sec. 3 P3 | "the nearest cancelling point lies through zero"; "crosses sigma = pi" | "(in angle coordinates)"; "crosses \|sigma\| = pi" | TC 6 |
| 10 | main Sec. 4.3 | "(an unreachable wrap, P3; ...)" | "(a wrap past \|sigma\| = pi, P3; ...)"; E P3 closing paragraph and P4 trained-model paragraph likewise ("reachable, but only through \|sigma\| = pi") | TC 6 |
| 11 | main Sec. 3 collective route; E Remark R | "the cancelling set {z(-1) = -z(+1)} is a linear subspace" | "the latents at which every channel cancels through zero, {z(-1) = -z(+1)} ... (W_out of full column rank)"; R gives the general set {W_out (z(+) + z(-)) = 0} | TC 9 |
| 12 | main Lemma 1; E T1 | "(equivalently omega_c, or row c of W_out)" | "(equivalently omega_c; in the signed class also row c of W_out)"; T1: "In the \|.\| class only the omega_c form is available" | TC 3 fix 1 |
| 13 | main Lemma 1; E T1 remark | "checked bitwise"; "hold bitwise" | "checked bitwise on CPU"; "hold bitwise in the CPU kernels used (... not of IEEE 754)" | TC 3 fix 2 |
| 14 | E T1 check list | "MapEM with q0_pos and k0_pos reflected: identical." | adds that the control moves logits by only 4.7e-5 (q0/k0 at init), so the sub-check is uninformative as run | TC 3 fix 3 |
| 15 | E T1(i); main Lemma 1(i) | "iff every row of S is sign-coherent" | stated on Theta (omega_c = 0 edge noted); main: "every row of the angle table" | TC 3 |
| 16 | E B0(a) | "which is P1(a) and covers every model in the paper" | covers every observation-step-free model and a map code for every \|.\| arm; the full arms' trained tables (V = 19/21 > r) are not covered | TC 2 |
| 17 | E B0(b) | "a step cannot be carried from one side of zero to the other by a small parameter change" | "the step is never negative, so no path moves one token's angle in a channel through zero while another's stays put (omega flips a whole channel; T1)" | TC 2 |
| 18 | E A0 notes 2, 3 | "Real s gives a decay ... in (0, 1]"; "Delta_t = softplus(.) >= 0" | "Real s > 0 with a non-negative rate ..."; "Delta_t > 0 (mamba3.txt:154; softplus in the implementation)" | TC 1 |
| 19 | E A0 (moved paragraph) | draft 5 Sec. 3 "with Phi_t = sum Delta_u, channel c ... exp(-i omega_c (Phi_t - Phi_{t'}))" | Phi_{c,t} per channel | TC 1 (Sec. 3 fix) |
| 20 | main Sec. 1 | "a rotation by a non-negative angle has complex eigenvalues" | "a rotation by theta has eigenvalues e^{+-i theta}, real and non-negative only at multiples of 2 pi" | TC 12 |
| 21 | abstract | "(Delta mod 2 pi / omega is the same model)" | "(Delta mod 2 pi/\|omega\| gives the same angles)" | TC 12 |
| 22 | main Sec. 3 Representability | "so P1(a) covers every \|.\| arm here" | "so every \|.\| arm can represent any code in which only the moves step"; E P1(a) likewise ("any code in which only the moves step") | TC 4 |
| 23 | E P5 | (a) unscoped; (d) "onto the wrap segment" | (a) "In the observation-step-free model"; (d) "minus the point (pi, pi)"; scope: moves +-1 only | TC 8 |
| 24 | E check-text slips | "m = 1..4; 39-56 of 64"; script labels; "each step takes about 1-5 min"; "draft 4 Sec. 4.2" | "for m = 2..4, 39-56 (m = 1 has none)"; "(labelled "(b)"/"(c)" in the script)"; "1.5-20 s"; "draft 6 Sec. 4.2"; header points to draft 6 | TC 13 |
| 25 | theorem numbers | uncited | Grazzi Thms. 1, 2, 3 (App. C.1 in E), Sarrof Thm. 2, Amid and Warmuth Prop. 2 (at tau = 0 in Sec. 5), Mamba-3 Sec. 3.2 (Sec. 1) | TC 12, 14 |

Not applied: TC 14's optional textbook reference for Kronecker / Dirichlet (E names the theorems; adding a reference entry
for an appendix-only use was judged unnecessary). E's "Corrections in v3" list (items 14-22) records all of the above.

## 2. Softplus (T item 2)

Replaced "untested representability" in: main Sec. 3 (P1(f), with the construction and its check), Sec. 6 ("its
representability is proved, its existence under training untested"), abstract and contribution 2 ("Both barrier classes can
represent the map exactly"), D4 (construction, check numbers, "on the torus m = 4 = r, so the same construction applies (not
built)"), E (new Proposition P1(f) with proof and check line; summary-table row). Main text and D4 state that existence under
training (frozen / trainable continuation) was not tested. Check: `docs/paper/review_draft5/theory_check_extra.py` (2), as
TC reports (min step 0.693, angle error 8.4e-15, logits 2.8e-11).

## 3. NUMBERS_AUDIT errors (NA 1-12), all applied

| NA | location | old | new |
|---|---|---|---|
| 1 | Fig. K1 caption (figure list); app. K | "(overall 0.507); the n-gram floor 0.576 is not drawn" | "0.513 (overall 0.509 on this stream); the n-gram floor (0.576, computed on another stream of the same generator) is not drawn"; K text adds the same note |
| 2 | Sec. 4.1 | "every other sign contrast holds at thresholds 0.02 and 0.1 and under accuracy >= 0.99" | "every other contrast in Table C10 holds at 0.02, 0.1 and accuracy >= 0.99"; C10 caption adds ROBUST torus 5/8 vs 8/8 (p 0.20) at 0.1 |
| 3 | Sec. 4.1 (ROBUST) | "(5/8 vs 8/8; post hoc)" | "(5/8 vs 8/8, p 0.20; a declared secondary)"; C13 says the column is a declared secondary |
| 4 | Sec. 4.1; B Gates; A5 | "the one-layer index model's 0.72-0.73 at p_plus 0.5"; B "0.717 / 0.676 / 0.825 (CANCEL)" | main "(0.730 at p_plus 0.5)" (GENERALITY, checked batch); B "0.710 / 0.677 / 0.824 on CANCEL's six fresh seeds (0.717 / 0.676 / 0.825 with its two pilot seeds; no independent check)"; A5 CANCEL row added |
| 5 | A4 dropout re-score | "declared secondary in the batches after 2026-10-04 ... changed no label in any batch cited here (each results file states it)" | lists the six batches that declared it, the 2026-10-04 audit for SIGN_MATCHED / TEXTWORLD, and that ACCOUNTS, TORUS_START, GAUGE, ROBUST were not re-scored (ROBUST: invalid at 2 layers), FACTORIAL declared but did not report |
| 6 | A1, A3, A5 | ROBUST missing | A1 layers row adds ROBUST torus 2 layers d 256; A3 kernel row adds ROBUST, new "recipe variants" row; A5 ROBUST row (seeds 1400-1407, Amendment 1, pilot 1490) |
| 7 | Sec. 2 Protocol | "a pre-launch gate by the authors and an independent audit after the run" | "authors' pre-launch gate, audit after the run"; A5: "audit after the run (2026-09-30), which reproduced its probe (no file records it as independent)" |
| 8 | C9 caption; `GAUGE_RESULTS.md` | "87.5% of revisits are backtracks" | "89% ... (138,800 of 155,456 on the eval stream; 87.5% in the pre-launch audit's sample)"; two-line CORRECTED banner at the top of GAUGE_RESULTS.md (89.3%) |
| 9 | app. H CARoPE | "any return interval advances each channel by a value in (0, 4]" | "the shortest return advances each channel by a value in (0, 4]" |
| 10 | abstract | "it shrinks at three times the learning rate." | "... and, in accuracy, in the larger model." |
| 11 | abstract | "against 8 of 8 for signed steps." | "against 8 of 8 for signed steps at the default recipe (6 of 8 at a third of the learning rate)" |
| 12 | figure list | "Appendix tables C1-C12 (and C9b)" | "C1-C13 (and C9b)" |

## 4. Framing (SY 5; MR W2)

| location | draft 5 | draft 6 |
|---|---|---|
| abstract | "A per-channel identity shows the barrier removes, for an inverse pair, the through-zero route to cancellation, not the function. On the ring signed steps take that route collectively; ..." | "The barrier removes the through-zero route by which signed steps cancel an inverse pair, but that does not explain the failure: with moves +1/+2, where no step need cross zero, \|.\| still solves 2 of 8 (signed 5 of 8; unmeasured)." |
| Sec. 1 | "We identify a route to the map that the barrier removes; we do not explain why barrier steps fail from small starts, or with +1/+2 moves ..." | MapFormer credited for the end-state code (Figs. 3d, 9b); "signed steps reach it collectively, but losing this route does not explain the barrier's failure (Sec. 4.5)" |
| contribution 3 | "for an inverse pair, a route to cancellation, not the function. A registered test ... was inconclusive" | "for an inverse pair it removes the through-zero route, not the function. Its learnability prediction was not confirmed" |
| contribution 4 | "*A route, on the ring*: signed steps leave the barrier's start through zero ..." | "*The route signed steps use* (ring): through zero, collectively via their latents" |
| 4.3 heading / closing | "a route through zero"; "So the barrier removes a collective route through zero (registered); why \|.\| fails ... is not separated" | "the route signed steps take"; "Why \|.\| fails from a small start (...) is open"; "MapFormer described the end state; this is the causal test" |
| 4.5 closing | "Nothing here suggests the cost needs an inverse pair, and the barrier account does not explain \|.\|'s difficulty on +1/+2" | "So losing the through-zero route does not account for \|.\|'s difficulty here; Lemma 1 is a statement about representation only" |
| 4.1 | "In every cell the \|.\| runs that solve do so without the wrapped code" | "In every cell \|.\| shows little wrapped code, even at lr x3 with 4/8 solved (unweighted arm share ...; a declared secondary)" (the share is an arm mean, unweighted) |
| Sec. 6 | "The mechanism is shown on the ring only, with its \|.\| side unresolved" | "The route is shown on the ring only and does not explain the \|.\| failure" |

## 5. Scoping (SY 6; MR)

| item | draft 5 | draft 6 |
|---|---|---|
| 4.1 heading | "Barrier parameterisations do not learn the map" | "... rarely learn the map within budget" |
| 4.2 heading | "The \|.\| solution exists and survives training" | "The \|.\| solution exists: kept on the ring, re-found on the torus" |
| 4.4 heading | "The start lever does not transfer to the torus" | "The start lever fails on the torus within 900 epochs"; contribution 4 "within 900 epochs" |
| 4.6 heading, contribution 5 | "fail these maps" | "(within budget)"; "do not learn these maps within budget" |
| ring claim | "the ring evidence is the SOLVED criterion" | adds "they rarely reach an exact map"; abstract "on the ring \|.\| reaches 0.95-0.97 accuracy but rarely an exact map"; contribution 1 "rarely reach an exact map" |
| scale | "Exact \|.\| solutions exist at a trained scale"; "the weights stay within ~5x" (ring only) | abstract "built \|.\| solutions"; 4.2 "weights within ~5x of the signed model's" (ring 5x, torus ~2.7x); "posed near a trained scale" |
| unmarked declared secondaries | 4.1 wrapped-code share; 4.2 usage-weighted 99% -> 98%; 4.5 0.92 sign-coherent | "a declared secondary"; "a registered readout of ACCOUNTS T1"; "a declared secondary" |
| Sec. 2 protocol | "Every contrast was pre-registered with its decision rule" | "Every verdict was pre-registered with its decision rule (post hoc readouts are marked)" |
| lr scope | "within our budgets (300-900 epochs)" | abstract / Sec. 2 / Sec. 6 add "and learning rates"; 4.1 adds "\|.\| SOLVED rises with the rate (0/8, 1/8, 4/8), and all other comparisons use the default rate, 1e-3"; Sec. 6 names the missing per-arm lr sweep |
| lr / 3 | "at a third of the learning rate (0/8 vs 6/8, Fisher p 0.007)" | "on SOLVED at a third of the learning rate (...; accuracy p 0.089)" (MR W6) |
| cross-batch torus (RA 5) | "torus 0/8 vs 8/8 (-0.120) and 1/8 vs 8/8 (-0.104)" beside SIGN_MATCHED | adds "(unlike the torus sign batch: fused attention kernel, observation-step-free classes; appendix A3)"; A3 gets a "Cross-batch torus comparisons" paragraph |
| 4.3 (iii) | "The sum, not only the size" | "Size alone does not ... sum and size are only partly separated, appendix C12" (MR Q9) |

Note on the brief's RA 5 instruction ("at 'a quarter of the one-layer gap' say the batches differ in --fast-attn and model
class"): checked against `run_robust.sh` and `run_generality.sh` -- ROBUST's torus cell and GENERALITY both use
`--fast-attn` and the AO4 classes, so that comparison does not cross kernels or classes. The kernel / class caveat is placed
where it does apply (SIGN_MATCHED vs GENERALITY / TORUS_START, Sec. 4.1, A3), and A3 states that the quarter comparison is out
of batch but not across kernels.

## 6. CoPE fragility and ACCOUNTS T1 (SY 7-8; RA 2-3)

- Sec. 4.6: "On the line CoPE still fails at the registered criterion (1/16 vs 0/16 on the ring) ... removing the wrap does
  not make CoPE learnable. At a looser threshold half the line runs pass" -> "On the line CoPE solves 1/16 (ring 0/16) ...
  For CoPE the loss criterion is fragile: 5/16 line runs score >= 0.9997 held out yet miss the (dropout-on) loss cut, and at
  held-out accuracy >= 0.999 the line reads 5/16 vs 2/16 (post hoc), so whether the wrap adds to CoPE's difficulty is
  unmeasured." App. H and a new C10 note carry the numbers and the other single-run cases (RA 3, 4).
- T1 bit-identical (RA 2): Sec. 4.2 ("a registered readout of ACCOUNTS T1, whose retrained runs are bit-identical to these:
  determinism, not replication"), A5 ACCOUNTS row, D6, E P2 status.

## 7. Citations (SY 10; PA), each checked first-hand

| citation | where | what was verified (file:line) |
|---|---|---|
| Shakerinava, Khavari, Ravanbakhsh, Chandar, ICLR 2026, arXiv 2603.01959 (new) | Sec. 5 | AUSSM transitions exp(i Delta(x) Lambda(x)), Delta >= 0, Lambda signed (`papers/new_2026-10-16/2603.01959/paper.txt:179-183`); single-layer AUSSM learns the six Abelian groups (`:704-712`); "not expressivity but optimization ... solutions exist in weight space, yet standard training does not reliably reach them"; near-solution init (`:724-731`) |
| Helfrich, Willmott, Ye, ICML 2018 (new) | Sec. 5 | "D is fixed throughout the training process. The number of -1s on the diagonal of D ... is considered a hyperparameter" (`1707.09520/paper.txt:160-169`); Cayley near -1 "very slowly, if at all" (`:118-128`). Worded as "fixed during training, its count of -1s a per-task hyperparameter" (the paper fixes D; it does not prove D unlearnable) |
| Wang and Wang (new) | Sec. 5 novelty sentence | RoPE commutant aI_2 + bJ, GL(1, C) per plane (`wang2026_gauge/paper.txt:277-280`); no further symmetries (Thm. 24, `:1436-1442`). **Year: the PDF header reads "Proceedings of Machine Learning Research 282:622-663, 2025"; PA and the brief say 2026. Cited as 2025; verify before submission.** |
| Tran et al., ICML 2026 (PMLR 306), arXiv 2606.17830 (new) | Sec. 5 | RoPE reduces the symmetry group (abstract, `2606.17830/paper.txt:128-137`); G_RoPE with H(d_h) abelian commuting with R_n (`:352-400`); venue line `:163-164` |
| MapFormer v4 Table 2 | Sec. 4.6, app. H | CoPE 1L 0.63 +/- 0.21 (1D IID), 0.65 (2D IID); 4L 0.88 / 0.76; MapWM 1.00 (`papers/txt/mapformer.txt:436-560`); attribution to the sign (`:584-586`, `:1701-1703`) |
| MapFormer v4 Figs. 3d, 9b | Sec. 1, Sec. 4.3, E Remark R | "opposite actions map to opposite directions" (`:650-655`); cos(Delta_left, Delta_right) = -1 (`:2115-2158`) |
| Grazzi et al. Thms. 1, 2, 3 | Sec. 1, Sec. 3, E T1(ii) | Thm. 1 parity needs an eigenvalue not in [0, inf); Thm. 2 mod m (m not a power of two) needs a non-real eigenvalue; Thm. 3 rotations for Z_m (`papers/txt/grazzi.txt:312-334`, `:483-496`) |
| Sarrof et al. Thm. 2 | Sec. 1 | "No SSM satisfying nonnegative can recognize PARITY" (`papers/txt/sarrof.txt:314`) |
| Amid and Warmuth Prop. 2 | Sec. 3 P4, Sec. 5 | w = q_tau(u) ~ \|u\|^{2/(2-tau)}, \|u\| at tau = 0 (`papers/new_2026-10-14/2002.10487/paper.txt:476-490`) |
| Khavari et al. Thm. 4.4 (fixed sentence) | Sec. 5 | S4D + Mamba layers cannot solve parity (`papers/new_2026-10-15/2508.07395/paper.txt:219-224`). Old: "Khavari et al. (2025) add that ... none of these theorems covers rotations by non-negative angles" (no antecedent; S4D's fixed rotations are covered). New: "In diagonal SSMs, parity needs input dependence and negative eigenvalues in the same layer (Khavari et al. 2025, Thm. 4.4); none of these theorems restricts an input-dependent rotation by a non-negative angle." |
| Grazzi sentence (fixed) | Sec. 1 | Old: "transitions with real non-negative eigenvalues cannot track parity or count modulo m". New: "real non-negative eigenvalues cannot track parity (...Thm. 2; ... Thm. 1), and real eigenvalues cannot count modulo m for m not a power of two (Grazzi et al., Thm. 2)" |
| Novelty sentence (scoped) | Sec. 5 | Old: "To our knowledge neither the relative-sign gauge nor the loss of the through-zero route for inverse pairs has been stated". New: "With fixed frequencies the parameter symmetries of rotary attention are per-plane complex scalars, with no reflection (Wang and Wang 2025; Tran et al. 2026); to our knowledge the reflection a learnable step adds (Lemma 1), the loss of the through-zero route for inverse pairs, and the learnability cost of a non-negative rotary step have not been stated." |
| LRU positioning | Sec. 5 | adds "they are input-independent, so by Lemma 1(ii) their sign is a convention" (PA 3.1; MR novelty) |
| Selective RoPE, Mamba-3, CARoPE, CoPE | Sec. 1 | author citations moved into Sec. 1 (the Related-work list sentence was cut) |

Not added (optional in PA): Howe 2026, Karuvally et al. 2025, Ebrahimi and Memisevic 2025, Lezcano-Casado 2019, Hayes et al.
2023, Ran-Milo et al. 2024 (word budget). Liu et al., Hahn and Rofin and Yang et al. are now cited in appendix K only (the
reference preamble says so).

## 8. New floor (T item 8)

Draft 5 cites no T = 128 torus number (its torus numbers are all T = 1024), so nothing in the main text needed the floor.
Appendix B gains two rows and a paragraph: torus T = 128 on PAPER2X2's eval walks, blank 0.507 / n-gram 0.598 / retrace
0.934 (per seed 0.9277-0.9455; no wrap-only revisits), with index RoPE 0.805 below retrace and path r = 2 0.971 just above
(`docs/audits/2026-10-16/paper2x2_T128_floors_out.txt`; `sign_floors.py` with `FLOOR_T=128`, n-gram fit on T = 1024 walks);
and ROBUST's torus own-stream floors 0.509 / 0.578 / 0.844, every run above retrace (lowest \|.\| 0.910;
`robust_floors_out.txt`). The outline's contribution 1 quotes +0.243 at T = 128; it should carry the 0.934 floor (not
edited here).

## 9. ROBUST rows (T item 9; NA 6)

A1 (layers row), A3 (kernel row; new "recipe variants" row), A5 (ROBUST row). C13 also gains: lrd3 accuracy not detected;
the matched-seed |.| trend 0/8 -> 1/8 -> 4/8; that the lr cells also change lr x wd and the cosine floor; own-stream floors
and strata; that the loss-0.1 column is a declared secondary.

## 10. Figures (T item 10; SY 9)

Main-text figures renumbered: Fig. 1 = `fig_loss_curves` (new; Sec. 4.1, the headline contrast), Fig. 2 = `fig_wrap_share`
(usage-weighted redraw; draft 5's Fig. 1), Fig. 3 = `fig_mechanism` (shading fix; draft 5's Fig. 2), Fig. K1 =
`fig_factorial_strata` (floor-line fix), each with "[CAPTION FROM FIGS AGENT]" and draft 5's caption facts kept beside it
for the figures agent. `figs/` was not edited. Sec. 4.1 quotes the unweighted shares, labelled, with
"[NUMBERS FROM FIGS AGENT]" for the usage-weighted ones. A4 describes the unweighted share's denominator and why it misleads.

## Other changes

- Title unchanged. SY A1 / MR W1 tie a retitle to the per-arm learning-rate sweep, which has not run.
- Sec. 2 Protocol: the independent check is described in A5 (a separately launched, read-only AI code-review session given
  the pre-registration and code, not results; independent of the session that wrote the code, not of the project) (MR W11).
- Sec. 4.2: the lr 1e-4 rationale (pilot knocked out the signed control at 1e-3) is in D5 only.
- Reference preamble: PMLR added; process paths removed.

## Not applied, and why

- SY A1 (per-arm lr sweep) and SY 11 (cut to 4 pages of layout) are a GPU batch and a venue decision, outside this text pass;
  the main text is at 3,497 words (two tables, three figures), still above a 4-page layout.
- MR W4's request to add the solved |.| runs' usage-weighted share to the text waits for the figures agent's numbers.
- MR W10 (r = 8 \|.\| torus arm), W5 (exp(.) arm): experiments, listed in Limitations.

## Rule-28 end-to-end read (draft 6 main text, after all edits)

Read from title to figure list in one pass. Checked: every number against draft 5 or the review's corrected values; every
section cross-reference (Secs. 3, 4.1-4.6, appendices A3, B, C10, C12, C13, D, E, H, I, K resolve); figure numbers (Fig. 1-3
in order of first mention: 4.1, 4.1, 4.3); every author-year in the text has a reference entry and every entry is cited in the
main text or appendix K; status labels (registered / declared secondary / post hoc / descriptive) on every non-primary
number. Fixed during the read: author citations for Selective RoPE / Mamba-3 / CARoPE / CoPE restored in Sec. 1 after the
Related-work list was cut; "trained and tested at the same length" restored in Sec. 2; "Prop. 2 at tau = 0" in Sec. 5;
"(0.730 at p_plus 0.5)"; "no cancelling channel wrapping" (was ambiguous "none wrapping"). Remaining known weak spots: the
3-word margin under 3,500; Fig. 2's numbers pending; the Wang and Wang year.

## 2026-10-25: G-3 (SSM, blocks A and A2) folded in

- Title: "parameterised to be non-negative" -> "parameterised as |s| or softplus(s)". G-3 block A2 (`G3_SSM_A2_RESULTS.md`,
  registered) found NO SIGN COST for an exp(s) (LRU-style) step in a one-layer SSM at lr 1e-3 (6/8 vs signed 7/8), so the
  old title generalised past the evidence; |.| and softplus paid at both SSM rates.
- Sec. 4.1: new paragraph "Without softmax" (|.| 0/8 vs 8/8 at 3e-3 and 0/8 vs 7/8 at 1e-3; softplus 1/8, 0/8; exp diverged
  at 3e-3, no cost at 1e-3, wrapping on past-pi channels, post hoc; exp 1/8 in attention). Sec. 5: one clause after the LRU
  sentence. Limitations: "no recurrent, SSM or exp(.)-type (LRU) baselines" replaced by the SSM test's scope and the exp
  result. Word count +~150 (above the 3,500 target; trim before submission). MR W5 (exp(.) arm) is now answered in the SSM.

## Fold of EXPTORUS (2026-10-26)

- EXPTORUS (`EXPTORUS_RESULTS.md`, registered + Amendments 1-2): in attention on the observation-step-free T = 1024 torus,
  each step at its best of lr 1e-3 / 3e-3, exp 0/8 and softplus 0/8 vs signed 8/8 (Holm p 0.0003 on SOLVED; BOTH PAY;
  budget-scoped; exp partly pilot-informed). Sec. 4.1 "Without softmax": exp pays in attention on the ring and the torus,
  escapes only in the SSM on the ring. Sec. 5 (LRU sentence) and Limitations updated (mixer, task and start not
  separated). Title unchanged (|s| or softplus(s) remain the forms that pay everywhere). Word count 5,296 -> 5,396.
