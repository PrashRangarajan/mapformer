# Draft 4 vs draft 3: changes and number sources (2026-10-15)

Draft 4: `docs/paper/DRAFT4_workshop.md`. Draft 3 (`DRAFT3_workshop.md`) and the v1 theory appendix (`APPENDIX_theory.md`) are
unchanged; the refreshed appendix is `docs/paper/APPENDIX_theory_v2.md`. Inputs: `MOCK_REVIEW_draft3.md` (all 14 no-experiment
fixes, weaknesses W1-W10), the caller's seven required items, `CLAIM_MAP_clock_map.md` ("Do not claim", through 2026-10-15),
the results files listed below, `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`, `docs/audits/2026-10-12/ring_dial_audit.md`
(M1), `docs/paper/figs/FIGS_NOTES.md`, `docs/audits/2026-10-14/theory_review/*_out.txt`, `GAUGE_PREREG.md`. No GPU, no
experiment, no new computation: every new number below is read from an existing committed output.

Length: abstract to end of Limitations 3,611 words by `wc -w` (tables, formula tokens and the 115-word GAUGE placeholder
included; ~3,490 without the placeholder). Draft 3 was ~3,650 on the same count. Sections 4.6 (depth) and 4.7 (clock side) of
draft 3 moved to appendices K and J (~340 words), which paid for the new material.

## 1. The caller's required items

| # | requirement | what was done | where |
|---|---|---|---|
| R1 | "Non-negative" as a property of the parameterisation; Theorem 1 honest | Abstract and Sec. 1 now say the step *value* is harmless (Delta mod 2 pi / omega is non-negative and the same model) and the tested object is a *barrier parameterisation* (|s| folds, softplus flattens; gradient descent cannot carry a step through zero). Sec. 2 defines "barrier parameterisations". Theorem 1 -> "Lemma 1 (relative-sign gauge)", stated as an elementary identity whose consequences are the content; (iii) now reads "removes a route to cancellation, not cancellation" (the through-zero code and the wrapped code are the same function). Appendix v2 adds Remark B0 (values vs parameterisations, with a rank caveat: the entrywise remainder stays rank r only when <= r tokens step, P1(a)) | draft Secs. Abstract, 1, 2, 3; appendix B0, T1 |
| R2 | Sticky wall with data | 4.3 bullet "A competing account, the wall": 44% of AO_absmag channels end with |a(+1)| <= 0.05 rad vs SO_absmag 17%, AO_mid 0.2%, AO_high 1.4%; consistent with P4; three unknowns stated (start, usage weighting, gradient at the wall); closing sentence says the failure route (P3 vs P4) is not separated. Fig. 2 caption: 0.035 of AO_absmag's 0.04 is collapsed near-origin channels | draft 4.3, Fig. 2; appendix P4 "Trained models" |
| R3 | |.| side loose | 4.3 "What fits loosely": rescued models cancel in 23% (AO_high) / 34% (SO_high) of channels; per-channel flow predicts 28%, so per-channel capture is a live alternative (not compared channel by channel); |default| start sums 0.01-6.58 rad, 5% past pi, and the 20 non-collapsed channels starting in [pi, 3pi/2) end with none cancelling; capture falls gently with distance from 2 pi (0.34 -> 0.08). Sec. 1 now says "we do not fully explain why barrier steps fail from small starts" | draft 1, 4.3; appendix R |
| R4 | Protocol statements | Sec. 2 Protocol rewritten: every main-text batch but SIGN_MATCHED (torus, Sec. 4.1) had an independent pre-launch code check, and any pilot used outside seeds; in the appendix, TEXTWORLD (reused pilot seeds) and DYCK_MDEPTH (predates the checks) are named. Fresh-seed text-world numbers quoted. See Section 3 below for the audit trail | draft 2, 4.1, Limitations, appendix K |
| R5 | Title / abstract scope | Title names the barrier, "one-layer models, synthetic maps"; abstract says "one-layer rotary models", "synthetic navigation maps", "within our budgets (300-900 epochs, one recipe per task)". "Learn only from" -> "learn reliably only from". "Exactly those" removed (abstract now "the through-zero route to cancellation, not the function"). "0-3 of 8" -> "0-4 of 8 seeds per batch (0-2 wherever a registered contrast fires)", which includes the unmeasured 4/8 at p_plus 0.9 | title, abstract |
| R6 | Appendix v2 | New file. Drift / weight-decay account under P2 deleted (P2.1 kept as a one-line mathematical remark plus a Status paragraph citing ACCOUNTS T1); T2 recorded as SUPPORTED in Remark R; T4 recorded under P7; every v1 "Checked" pointer kept; summary table and corrections list updated | `APPENDIX_theory_v2.md` |
| R7 | GAUGE placeholder | Sec. 4.5: registered design from `GAUGE_PREREG.md` (moves +1/-1 vs +1/+2, AO signed vs |.|, RoPE on +1/+2, n = 8, seeds 1300-1307), the fixed reading, and "[RESULTS PENDING]". No result invented; the pilot (`runs/gauge_pilot`) was not read | draft 4.5; appendix T1(ii) "Pending test" |

## 2. Mock review Part 3.1 fixes (no experiments)

| # | fix | status in draft 4 |
|---|---|---|
| 1 | Scope title and abstract; "reliably only from"; "0-4 of 8"; qualify "exactly"; scope sentence | done (R5). The mock's suggested title was adapted to name the barrier and the scope |
| 2 | Reframe "sign" as "barrier at zero" | done (R1) |
| 3 | Rewrite representability paragraph: P1(a) with W_in annihilating observations covers every |.| arm (M1); "given table" as a side remark; Corollary (i) "as real numbers, same latent" | done: Sec. 3 "Representability" and Lemma 1(i); appendix P1(a) "The full |.| arms" and T1(i) scope note |
| 4 | Theorem 1 -> Lemma 1; drop "exact core"; present as predictive; say untested | done; prediction stated as Lemma 1(ii) with the +1/+2 case; test pending (4.5) |
| 5 | Protocol sentence: text world pilot seeds (fresh 6/6 vs 0/6, p 0.0022); SIGN_MATCHED no pre-launch check; recency n = 16 | done: Sec. 2 Protocol; fresh-seed numbers in appendix K; recency n = 16 in appendix J (the clock side left the main text, so the n is given there) |
| 6 | Mechanism loose ends: 23% / 34%; sums 0.01-6.58; 44% collapse; "identify the reason" -> "a route"; label 4.4 account | done (R2, R3); Sec. 1 "We identify a route ... we do not fully explain"; 4.4 "An untested account: ..." |
| 7 | "Stable in task terms" ring only; torus lr 1e-3 pair (6/8) | done: 4.2 ends "stable in task terms on the ring and re-found from nearby on the torus"; lr 1e-3 descriptive 6/8 (0.988) and the reason for lr 1e-4 (pre-launch pilot) stated. Abstract keeps "keeps (ring) or re-finds (torus)" |
| 8 | Move 4.6 and 4.7 to appendix; drop contribution 6; Liu C5/C7/C8; index long-gap at blank floor | done: appendices K and J; contribution 6 dropped; Liu sentence now cites both directions (C_4 below 90% at 1-15 layers, C_2 and C_6 unreliable; C_5, C_7, C_8 >= 98.5% from 3 layers; checked against `liu2023.txt:1717-1745`; the mock's ">= 99%" is 98.5 for C_8 at 4 and 10 layers); "at or below the blank floor" added. Related work keeps a two-sentence "Depth" paragraph pointing to K |
| 9 | Bring appendix E up to date | done as a new file (R6); header and section references updated (P1(a) -> draft 4 Sec. 4.2) |
| 10 | Presentation: define terms first; rename angle sum; relabel Fig. 2 panels; collapsed channels in caption; cite Table 2; name the floor in Fig. 3 | Setup moved before Theory and now defines MapWM, Theta, |.|, softplus, AO, inverse pair, through zero / by wrapping, wrap-only, SOLVED; MapEM appears only as "the MapFormer-EM variant"; "small-beta potential" replaced by a plain description in the main text and defined in appendix R; LOST defined in appendix plan D. Angle sum renamed sigma (draft and appendix P3, P5, summary; P7's CoPE sigma disambiguated). Fig. 2 caption panels described in words ((a)-(d)); **the PNG's panel titles still show arm codes: `fig_mechanism.py` needs relabelling (not done: outside the three files this pass may write)**. Table 2 cited in 4.3. Fig. 3 is now appendix Fig. K1; caption names the drawn floor as the averaged per-stratum best-constant floor (`fig_factorial_strata.py` l.21 draws `mean(floor)`, not exactly 0.507) and says the n-gram floor is not drawn. Fig. C1 cited as "appendix Fig. C1". `FIGS_NOTES.md` still calls the mechanism figure "Figure 3" (not edited) |
| 11 | Softplus initialisation-confound caveat | done: Sec. 2 (softplus(0) = 0.69, every token starts at the same step; `model_sign.py` l.43-53) and 4.1, Limitations |
| 12 | Remove internal process language | done: header reduced to one line marked for deletion; "one session of search" and "~8 h per torus run" removed; appendix plan E no longer records correction history (v2 keeps it in its own header and corrections list, where it belongs) |
| 13 | Related work: TEM-t; Dale's law (Cornford 2021; Li et al. 2023); CoPE construction refutes MapFormer's claim; verify author lists | TEM / TEM-t added (Integrators and position). Dale's law sentence added. MapFormer's CoPE statement quoted and corrected in 4.6 ("at least on the ring"; verified at `papers/txt/mapformer.txt:585-586, 1701-1703`). Author lists verified first-hand from local PDFs this pass: Woodworth et al. (Woodworth, Gunasekar, Lee, Moroshko, Savarese, Golan, Soudry, Srebro), Orvieto et al. (Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De), Hahn and Rofin, Amid and Warmuth, Mamba-3 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu). Li et al. 2023 = P. Li, J. Cornford, A. Ghosh, B. Richards, "Learning better with Dale's law: a spectral perspective", NeurIPS 2023 (from the NeurIPS / MLAnthology listing via web search; its claim, spectrum at initialisation rather than sign constraint, is abstract-level from memory: **verify before submission**). Still unverified: Cornford 2021 author list, Khavari 2025 author list, GridPE, Hahn-Rofin venue (ACL 2024) |
| 14 | CPU re-analyses from stored artefacts (thresholds 0.02 / 0.1 and accuracy criterion; channel-by-channel capture vs 23%; zero-collapse across all |.| arms) | **not done.** This pass was limited to writing three files and no new computation, and rule 7 requires a committed script behind every number. The draft states each gap instead: SOLVED threshold sensitivity "not shown" (4.1, Limitations); channel-by-channel capture "not compared" (4.3); the collapse is reported for the MECHANISM arms only (existing output). All three remain cheap post hoc CPU checks for the next pass (label post hoc; write predictions first for the wall certificate) |

## 3. Weaknesses answered in the text (W1-W10)

| W | answer in draft 4 |
|---|---|
| W1 generality | Title, abstract and Limitations scope to one layer, synthetic maps, one recipe, our budgets; Limitations lists the unvaried learning rate, weight decay from scratch, optimiser and depth; the descending runs (3/8 torus |.|) are stated in 4.1 |
| W2 mechanism ring-only, loose | 4.3 "What fits loosely" (R2, R3); 4.4 account labelled untested; Sec. 1 claims "a route", not "the reason" |
| W3 thin theory, wrong variable | Lemma 1 framed as elementary; barrier reframing (R1); Lemma 1(ii) made a registered prediction (4.5) |
| W4 representability muddled | Sec. 3 Representability rewritten (P1(a) covers every arm, M1 construction); Corollary (i) qualified |
| W5 ring is a loss threshold | 4.1 says so and that threshold sensitivity is not shown; the torus carries the accuracy evidence (-0.177, -0.120, -0.104) |
| W6 4.6 / 4.7 dilute | moved to appendices K and J; Liu quoted in both directions; floor stated |
| W7 rank slack on the torus | Limitations: signed cancelling code needs one latent dimension per inverse pair, the wrapped code also the 2 pi k / omega offsets; per-head rank decides torus learnability for signed steps (CLAUDE.md rank rows); uncontrolled |
| W8 narrow baselines | Limitations: no recurrent / SSM baseline, no exp(.) (LRU) parameterisation, CARoPE / CoPE at one layer; softplus confound |
| W9 presentation | as fix 10 |
| W10 protocol over-claim | as R4 |

## 4. Protocol audit trail (R4; checked first-hand)

- **SIGN_MATCHED (torus, Sec. 4.1).** `SIGN_MATCHED_PREREG.md` committed 2026-09-27 (29d44bd) "before any run", with a
  pre-launch gate written by the authors (`docs/audits/2026-09-27/gate_sign_matched.py`: parameter counts, Signed vs
  Vanilla_r4 on shared weights, Delta >= 0 at init, causal leak). No amendment and no independent check before launch;
  results committed 2026-09-28 (bbacd4a); an audit followed on 2026-09-30 (f17e934). Seeds 0-7; no pilot of this
  configuration is recorded, but the signed arm's recipe was taken from the rank line, where it had solved 8/8. The draft
  says "pre-launch gate by the authors; independent audit after the run" and does not call its seeds fresh.
- **TEXTWORLD (appendix K).** Amendment 1 (b10ad01, 2026-09-28) was written "after an independent audit, BEFORE any batch
  result was read", so it *was* independently checked before results; but `runs/textworld_pilot` s0/s1 are byte-identical
  to batch seeds 0-1 (CORRECTED block, 2026-09-30). Draft 4 quotes the fresh seeds (0.979 vs 0.505, +0.474, 6/6 vs 0/6,
  p 0.0022) first and the 8-seed and re-scored numbers second. The mock review's framing ("not independently checked")
  is corrected here: the defect is the reused pilot seeds, not the check.
- **Everything else in the main text** (RING_DIAL, RING_BUDGET, WRAP_EXISTENCE, MECHANISM, GENERALITY, WRAP_COPE, ACCOUNTS,
  TORUS_START; appendix: FACTORIAL, CLOCK_SPEED, CLOCK_SPEED_REP) cites an "Amendment 1 (independent check)" in its results
  header, with pilots on outside seeds (e.g. RING_DIAL seed 130, WRAP_EXISTENCE / GENERALITY seed 690; MECHANISM and
  TORUS_START had no GPU pilot). DYCK_MDEPTH (appendix F, 2026-09-26) predates the rule; it disclosed four seed-0 smoke runs
  with non-default flags and retrained seed 0 from scratch. Appendix F should say so when written.
- **Date of the rule.** The caller's brief and the mock review date independent checks "from 2026-10-11 / rule 29". In git,
  rule 29 entered CLAUDE.md on 2026-10-03 (e5850b4). The draft avoids dates; every main-text batch after 2026-10-03 has the
  check, so the statement is the same either way.

## 5. Other changes

| # | change | source |
|---|---|---|
| 1 | Sections reordered: Setup and terms (2) before Theory (3) so every term is defined before use; Table 1 folded into prose (its rows are the three Howard cases) | mock fix 10 |
| 2 | Contributions: 6 -> 5 (depth dropped); contribution 3 says "a one-line identity" and that its prediction is pending | mock W3, fix 8 |
| 3 | Remark R -> "The collective route (hypothesis)" in Sec. 3, tested in 4.3 | mock 2.4 |
| 4 | 4.1: second paragraph on strata folded into one sentence; torus 3/8 descending; ring p_plus 0.9 accuracy -0.046 | mock rows 2, 7; 2.4 |
| 5 | 4.2: usage-weighted 99% -> 98% now explained ("weighted by how strongly attention reads each channel"); parenthetical on learning rates replaced by the pre-launch reason | mock 2.4; `GENERALITY_PREREG.md` l.37-39 |
| 6 | 4.3: one narrative (Table 2 contrasts -> post hoc cancel route -> registered T2 -> loose ends -> one-sentence conclusion); F2 now "no detected effect" (was "costs nothing") | mock row 28, 2.4 |
| 7 | 4.6 (published methods): CoPE pooled 1 of 32 ring runs over three batches (post hoc pooling); MapFormer's CoPE statement corrected | mock row 34, fix 13 |
| 8 | Fig. 2 caption: "almost all through zero"; collapsed channels | mock row 48 |
| 9 | Appendix plan I: toy "fails often from the signed default (6/12 solve)" | mock 2.3 item 11 |
| 10 | Related work: "We give the same advice ... for a different reason" kept; "and we identify why" never used; LRU sentence kept with "needs them initialised near zero" | lit note Sec. 4 |

## 6. Number sources (new or changed in draft 4; unchanged numbers keep draft 3's sources, `DRAFT3_changes.md`)

| number(s) | file |
|---|---|
| 0-4 of 8 per batch; 0-2 where a contrast fires. Cells: torus Abs 0/8, Pos 1/8 (`SIGN_MATCHED_RESULTS.md`); torus AO 0/8 (`GENERALITY_RESULTS.md` E3), 1/8 (`TORUS_START_RESULTS.md` TS4); ring Abs 2/8, 1/8, 4/8 (`RING_DIAL_RESULTS.md`); ring AO 0/8, 0/8 (`WRAP_EXISTENCE_RESULTS.md` E3, `MECHANISM_RESULTS.md` M4); ring budget 3/8, 3/8 (`RING_BUDGET_RESULTS.md`, no in-batch signed arm). Fired: all except p_plus 0.9 (unmeasured) and RING_BUDGET (no contrast) | those files |
| from scratch AO 0-1 of 8 | GENERALITY E3, TORUS_START TS4, WRAP_EXISTENCE E3, MECHANISM M4 |
| torus |.| 3/8 DESCENDING at 900 epochs | `SIGN_MATCHED_RESULTS.md` caveats |
| softplus(0) = 0.69, init confound | `model_sign.py` l.43-53; `SIGN_MATCHED_PREREG.md` |
| M1: observation steps <= 1.7e-6, Abs steps min 5e-11, cos/sin 6.3e-4, circular mismatch 4.6e-4 | `docs/audits/2026-10-12/ring_dial_audit.md` l.46-53 |
| ring p_plus 0.9 accuracy -0.046 | `RING_DIAL_RESULTS.md` |
| torus AO accuracy gaps -0.120, -0.104 | GENERALITY E3, TORUS_START TS4 |
| torus continuation lr 1e-4 reason; lr 1e-3 6/8 (0.988) | `GENERALITY_PREREG.md` l.37-39; `GENERALITY_RESULTS.md` A |
| 44% / 17% / 0.2% / 1.4% near-zero forward angle; 0.035 of 0.042 same-sign near origin; init sum range 0.01-6.58 | `docs/paper/figs/fig_mechanism_out.txt` (post hoc) |
| 5% of channels start past pi (AO, AO_absmag) | `docs/audits/2026-10-14/theory_review/mech_channels_out.txt` (post hoc) |
| 20 non-collapsed channels in [pi, 3pi/2), cancel 0.00; 321 binned; < 5 beyond 3pi/2 | `docs/audits/2026-10-14/theory_review/mech_bins_out.txt` + `mech_bins.py` (bins printed only when n >= 5) |
| AO_high capture 0.34 within 0.25 rad of 2 pi, 0.08 at 2-3 rad | `docs/audits/2026-10-14/theory_review/mech_capture_out.txt` |
| 23% / 34% cancelling (AO_high / SO_high); 7% at start | `MECHANISM_RESULTS.md` secondaries; `fig_mechanism_out.txt` |
| per-channel flow 28% from U(1.7, 4.5); ~0.5 rad; 12% cross zero | `THEORY_REVIEW.md` item 6; `APPENDIX_theory.md` Remark R |
| CoPE 1 of 32 ring runs | GENERALITY 0/8 (seeds 700-707), WRAP_COPE CoPEF zeros init = the paper's, 1/8 (900-907), ACCOUNTS T4 ring 0/16 (1100-1115) |
| words, fresh seeds: 0.979 vs 0.505, +0.474, 6/6 vs 0/6, p 0.0022; re-score 0.993 | `TEXTWORLD_RESULTS.md` CORRECTED and AUDIT blocks |
| factorial long-gap 0.37-0.53, wrap-only 0.38-0.52 vs blank 0.507 | `FACTORIAL_RESULTS.md` secondaries |
| Liu medians: C_4 56.2-85.8 at 1-15 layers (99.3 at 16); C_5 >= 99.0, C_7 >= 99.4, C_8 >= 98.5 from 3 layers; C_2, C_6 below 90% at most depths | `papers/new_2026-10-14/2210.10749/liu2023.txt:1717-1745` (Fig. 9) |
| GAUGE design (no results) | `GAUGE_PREREG.md` (4c04bde) |

## 7. Do-not-claim check (claim map, binding)

Checked by reading and by grep (`replicat|drift|fragil|weight decay|exactly those|learn only from|identify the reason|Theorem 1|0-3`):
- Representability of non-negative rotations attributed to Grazzi / Mamba-3; never claimed; "cannot represent a map" absent.
- No "fragile under training" (the frozen probe is in appendix P2 only, labelled "nothing here says how training moves
  either code"); no drift claim; no weight-decay claim (appendix P2 Status records the withdrawal).
- "The wrap is CoPE's obstacle" stated as refuted; CoPE's precision stays "a plausible account" (the claim map's
  "(precision is)" is stronger than ACCOUNTS T4 supports; not followed).
- No general near-wrap design rule ("not a design rule", 4.4); no near-wrap CoPE rescue; no "only the start, not the
  constraint"; no double dissociation (appendix J); no "replicates Sarrof / Grazzi" (the only "replicates" is the
  in-batch TS4 sign cost).

## 8. Not done, or left for the author

Items marked DONE were closed in the finishing pass (Section 9).
- DONE: mock fix 14 (CPU re-analyses), Section 9.
- DONE: `fig_mechanism.py` panel titles and `FIGS_NOTES.md`'s "Figure 3" label.
- Source files the mock review flags as not bannered: `WRAP_EXISTENCE_RESULTS.md` (87% -> 18% drift secondary, now
  withdrawn by ACCOUNTS T1), `MECHANISM_RESULTS.md` ("Design implication", withdrawn by TORUS_START), `SIGN_MATCHED_RESULTS.md`
  and CLAUDE.md's sign row ("replication in a new regime"). Not edited here.
- Appendices A-D and F-I remain plans; J and K are written.
- DONE: citations (Li et al. 2023's claim, Cornford 2021 and Khavari 2025 author lists, GridPE, Hahn-Rofin venue); a
  reference list now ends the main text. Still not re-checked: the venues of Amid-Warmuth, Orvieto, Woodworth, Grazzi
  (marked * in the list).
- GAUGE (4.5) to be filled from its registered analysis only.

Sources (web, for one citation): [NeurIPS 2023 poster page](https://nips.cc/virtual/2023/poster/70321),
[MLAnthology entry](https://mlanthology.org/neurips/2023/li2023neurips-learning).

## 9. Finishing pass (2026-10-15; CPU only, no training)

Main text now 3,931 words by the same count (was 3,611): the re-analysis results below (~250 words) and rewrapped
citations. Every new number is post hoc and labelled so; the registered numbers are unchanged.

**Citations, checked first-hand** (downloads and metadata in `papers/new_2026-10-15/`, `SOURCES.txt` there):
| item | draft 4 said | found | change |
|---|---|---|---|
| Li et al. 2023 | "attribute the deficit to the spectrum at initialisation" | P. Li, J. Cornford, A. Ghosh, B. Richards, NeurIPS 2023: "the spectrum is more impactful than sign constraints on performance - though both contribute" (`li2023_neurips/paper.txt:79-80`) | "find the spectrum at initialisation matters more than the sign constraint, though both contribute" |
| Cornford et al. 2021 | (no list) | J. Cornford, D. Kalajdzievski, M. Leite, A. Lamarquette, D. M. Kullmann, B. A. Richards; ICLR 2021 poster (OpenReview eU776ZYxEpz; agrees with Li et al.'s ref. [10]) | reference entry |
| Khavari et al. 2025 | "the sign must be input-dependent in the same layer" | B. Khavari, M. Shakerinava, J. Khullar, J. Huang, F. Rivest, S. Ravanbakhsh, S. Chandar; ICML 2025 workshop (MOSS); claim: for diagonal SSMs a single layer must have both input dependence and negative eigenvalues | "in diagonal SSMs, input dependence and negative eigenvalues must occur in the same layer" |
| GridPE | "derives RoPE from a velocity-controlled oscillator" | B. Li, Y. Wu, N. Huang, W. Zhang, arXiv:2406.07049 (v1 2024, v3 2026): VCO grid-cell phase (Eqs. 1-3) + random Fourier features, reducing to RoPE in 1D; embeds given coordinates (images, point clouds) | "builds a position embedding for given coordinates from grid-cell velocity-controlled-oscillator models and Fourier features, recovering RoPE in one dimension" |
| Hahn-Rofin venue | ACL 2024, unverified | ACL 2024 long papers (2024.acl-long.800) | none |
Also checked against arXiv / OpenReview / Crossref metadata: every other entry's title and authors (TEM via Crossref, TEM-t
and Liu via OpenReview). `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md` Section 5 statuses updated.

**Mock fix 14, re-analyses** (`docs/audits/2026-10-15/reanalyses/`; predictions and decision rules in `PREDICTIONS.md`
written before any run; each probe checked on constructed cases first; outcome appended there):
- R1 `solved_thresholds.py` -> `solved_thresholds_out.md`. All 47 arms reproduce their registered SOLVED counts at 0.05.
  RING_DIAL p_plus 0.5 at loss < 0.1: 4/8 vs 8/8 (Fisher 0.077; does not fire); every other sign contrast, every Table 2
  contrast and T2 F1 fire at 0.02, 0.05, 0.1 and acc >= 0.99. CoPE line vs ring: 1/16 vs 0/16 registered, 8/16 vs 2/16 at
  0.1 (p 0.054), 7/16 vs 2/16 at acc >= 0.99. Draft: 4.1, Table 2 caption, 4.6, Limitations, appendix plan C.
- R2 `capture_by_channel.py` -> `capture_by_channel_out.txt`. The one-channel flow from each AO_high channel's own start
  predicts 0.329 cancelling vs 0.230 observed; kappa +0.086; near 2 pi flow 0.95 vs model 0.28; 59% of cancelling
  channels not captured by the flow. Draft: 4.3 second bullet and closing sentence.
- R3 `collapse_wall.py` -> `collapse_wall_out.md`. Collapse per move step in every barrier arm and signed control;
  ring AO arms: inherited from small starts (77-81% of near-zero starts end collapsed; signed 29-35%), read by attention
  (usage ratio 0.39-1.10), gradient into the wall for 51-57%: MIXED by the pre-stated rule. Draft: 4.3 third bullet.
  Gradient in eval mode (no dropout) on the ring eval stream; torus gradients not computed (collapse fractions only).
None needed training.

**Figure.** `fig_mechanism.py` panel titles now (a)-(d) in draft 4's words; the shading is labelled "barrier side: closed
to |.|"; re-rendered PNG/PDF; `fig_mechanism_out.txt` byte-identical. Draft Fig. 2 caption names the shading the same way.
