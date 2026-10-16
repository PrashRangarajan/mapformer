# Draft 5 vs draft 4: changes, reasons, word counts (2026-10-16)

Draft 5: `docs/paper/DRAFT5_workshop.md` (main text + references + figure list). Appendices A-D and F-K:
`docs/paper/APPENDIX_draft5.md` (new). Appendix E (theory): `docs/paper/APPENDIX_theory_v2.md` (unchanged). Draft 4 and its
change log are untouched. Inputs: `DRAFT4_workshop.md`, `READ_draft4.md` (judgement calls), `CLAIM_MAP_clock_map.md`,
`DRAFT4_changes.md`, `MOCK_REVIEW_draft3.md`, `figs/FIGS_NOTES.md`, `VENUES_2026-10-15.md`, and the results files each
number cites (CORRECTED / AUDIT blocks read first). No GPU, no training, no new analysis script: every number in draft 5 and
its appendix is copied from a committed output (the appendix's per-seed tables are reformatted copies; two "mean of the
per-seed values" in appendix B are labelled as such). Nothing committed.

## Word counts

Counted with `wc -w` from "## Abstract" to the line before "## References", tables, headings and formula tokens included
(the count `READ_draft4.md` used).

| section | draft 4 | draft 5 |
|---|---|---|
| Abstract | 239 | 186 |
| 1. Introduction | 373 | 338 |
| 2. Setup and terms | 385 | 372 |
| 3. Theory | 713 | 618 |
| 4.1 | 276 | 245 (incl. the 28-word ROBUST placeholder) |
| 4.2 | 228 | 210 |
| 4.3 | 778 | 419 |
| 4.4 | 107 | 103 |
| 4.5 | 325 | 289 |
| 4.6 | 221 | 197 |
| 5. Related work | 426 | 279 |
| 6. Limitations | 228 | 239 (incl. the 28-word ROBUST placeholder) |
| **main text** | **4,302** | **3,498** (3,442 without the two placeholders) |

(Section rows include their heading words; with the 3 words of the "## 4. Results" heading they sum to the total.) Whole
file: 5,918 -> 4,376 words (the appendix plan and appendices J, K left
the main file). `APPENDIX_draft5.md`: 9,159 words.

## The four approved decisions

**1. Title.** Draft 4: "A barrier at zero: rotary phase steps parameterised to be non-negative can represent a map but rarely
learn one (one-layer models, synthetic maps)". Draft 5: **"Rotary phase steps parameterised to be non-negative rarely learn a
map, even when they can represent it (one-layer models, synthetic maps)"**. It leads with the measured fact (rarely learn).
Tightened from the proposed wording because of decision 4: "parameterised to be non-negative can represent a map" asserts
representability for softplus too, which nothing in the repo shows; "even when they can represent it" makes representability
conditional (true for |.|, untested for softplus). Scope qualifiers kept in the title and in the abstract ("one-layer rotary
models ... synthetic maps", "within our budgets (300-900 epochs, one recipe per task)"). The barrier at zero is kept as an
account: abstract ("puts a barrier at zero that gradient steps cannot cross"; "Where no step need cross zero ... both classes
struggle"), Sec. 1, Sec. 3, 4.3's closing sentence.

**2. Cut to <= 3,500 words.** Moved or dropped, by section (destination in brackets; "dropped" only where no number was
carried):
- Abstract: "(0-2 wherever a registered contrast fires)" and "(7/8; 1/8 with them frozen)" dropped (READ cut list); last two
  sentences merged; the +1/+2 sentence added (decision 1).
- Sec. 1: the closing sentence shortened; contributions 3-5 tightened (decision 3).
- Sec. 2: the appendix-only protocol caveats (navigation in words reused 2 pilot seeds; Dyck predates the independent checks)
  -> appendix A5, F, G. The *wrap-only* definition -> appendix C11 (no longer used in the main text). "Floors ... given beside
  accuracies" -> "Floors ... are in appendix B" (see "Torus floors" below).
- Sec. 3: the score formula Re(conj(q~) k~ exp(...)) replaced by "depends on position only through exp(-i omega_c (Phi_t -
  Phi_t'))" (q~ no longer needed); the method-by-method case list folded into "the methods of Sec. 1"; the bitwise-check list
  -> "checked bitwise; appendix E"; P3's distance formulas and "both classes attain the same losses on the angle torus", and
  P4's "chatters within about one step" -> appendix E (already there); the one-channel flow numbers (12% cross zero, ~0.5 rad,
  28% of U(1.7, 4.5)) -> appendix C12 (READ: 4.3 repeated them).
- 4.1: threshold sentences folded into one (READ); the ring strata sentence (gap, sign-flip p 0.0078) -> appendix C11.
- 4.2: construction details (orthogonal latents, omega' = max(|omega|, 1)) -> appendix D2; the naive construction's cosines
  -0.987..-0.998 -> appendix D3.
- 4.3: the four "What fits loosely" bullets (~400 words) -> appendix C12 with every number; the main text keeps two sentences
  (23% cancelling, flow does not predict which, Cohen's kappa +0.09; collapse common, gradient check mixed) plus one on the toy
  (appendix I). Closing sentence shortened, still lists "or neither".
- 4.5: the laps reading kept but compressed (still labelled post hoc reading of a declared secondary); Table 2 caption shortened.
- 4.6: "Implemented faithfully, neither learns these maps at one layer" folded into "Faithful CARoPE"; the line-threshold
  numbers (8/16 vs 2/16, Fisher p 0.054; 7/16 at accuracy >= 0.99) -> appendix C10 and H, main text keeps the clause and the
  "unmeasured" reading.
- Sec. 5: GridPE and TEM (2020) sentences dropped (no numbers; their reference entries removed, nothing else cites them);
  the Howard repeat and the Sarrof / Grazzi / Mamba-3 parity repeat dropped (Secs. 1 and 3 give them); "Here, with inverse
  pairs sharing a channel ... no start we tried helped" dropped (repeats Secs. 4.3-4.4); Depth one sentence; CARoPE (Veisi
  et al.) and FoX (Lin et al.) now cited by name (they were in the reference list without an in-text citation).
- Sec. 6: "The torus sign batch had no independent pre-launch check" dropped from Limitations because Sec. 2 states it (READ
  cut list); every other scope statement kept, some compressed ("the signed cancelling code needs one latent dimension per
  inverse pair" folded into "less latent slack"). Added: softplus representability untested; the ROBUST placeholder.
- Figures and tables list: the "Appendix plan" and appendices J and K left the main file (now written in `APPENDIX_draft5.md`).

Kept in the main text on purpose (registered negatives and caveats): torus |.| 3/8 still descending; softplus initialisation
confound; p_plus 0.9 unmeasured; the 0.1-threshold failure at p_plus 0.5; 3x budget 3/8; 4.2's torus "perturbed and
re-found" with the lr 1e-3 descriptive 6/8 (briefly moved out during cutting, restored); 4.4's slowed signed model and
descending runs; 4.5 unmeasured and "not confirmed"; CARoPE ring unmeasured; CoPE line unmeasured; all Limitations items.

**3. Contribution 3.** Draft 4: "*What the barrier removes* (Lemma 1, a one-line identity): only relative signs within a
channel; for an inverse pair, a route, not a function. A registered test of its learnability corollary ... replicates the cost
on an inverse pair but cannot read the same-direction case: that task is harder for signed steps too." Draft 5: "*Which codes
the barrier excludes.* Lemma 1 (relative-sign gauge): only relative signs within a channel; for an inverse pair, a route to
cancellation, not the function. A registered test of its learnability prediction was inconclusive: moves +1/+2 were harder
for both classes (Sec. 4.5)." Lemma 1(ii) now ends "Sec. 4.5 does not confirm it"; Limitations "Lemma 1's learnability
prediction is untested at our power".

**4. Softplus scoped out of every existence / representability sentence.** Grep first: `grep -i softplus` over
`APPENDIX_theory_v2.md`, the results files and audits finds no construction or test of softplus representability (B0 only
says softplus cannot reach 0; `model_sign.py` docstring describes its initial step). Changes: abstract "Exact |.| solutions
exist"; contribution 2 "An exact |.| solution"; Sec. 3 Representability adds "Softplus cannot set a step to zero; whether the
softplus arms can represent the map, exactly or approximately, was not tested"; 4.2 heading "The |.| solution exists and
survives training" and "into |.| form"; Limitations "Softplus carries an initialisation confound and untested
representability (the existence results are for |.|)"; appendix D4 states that no softplus construction exists in the files.
Sentences about step *values* (abstract "Delta mod 2 pi / omega is the same model"; Sec. 3 "the last two cases represent the
same per-channel codes") are about values, not parameterisations, and are unchanged in substance ("per-channel" added).

## The other READ_draft4.md calls (text-only)

- **Torus floors in 4.1: not added.** `SIGN_MATCHED_RESULTS.md` reports no floor; `FACTORIAL_RESULTS.md` says "Floors on this
  stream: blank 0.507, best n-gram 0.576", but `analyze_factorial.py` l.70 prints them as "(RANK_MI stream)", and their source
  is `docs/audits/2026-10-05/score_rank_floors_out.txt`: stream seeds 10-25 (numpy seed 1234 + s), not the sign batch's walks
  (seeds 0-7; `eval_noise_refine` uses 1234 + run seed). So 0.507 / 0.576 cannot be confirmed for the sign batch's stream and
  were left out, as instructed. Found while checking: (a) the best-constant floor *was* computed on the sign batch's own walks,
  per seed, in `SIGN_MATCHED_STRATA.json` (0.501-0.512, mean 0.506), so the blank floor is confirmed; the n-gram is not;
  (b) the same floors file has a **retrace** predictor (a map-free rule) at **0.843** at T = 1024 (0.833-0.848 over 16 stream
  seeds), above the torus |.| mean 0.821 and softplus 0.781. Printing only the blank floor beside 0.821 would understate the
  floor, so 4.1 prints none; appendix B gives all three with their streams and says so. **Recommended (CPU, minutes):** compute
  the n-gram and retrace floors on the sign batch's own walks and then add "(floors: blank 0.51, retrace ~0.84)" to 4.1.
- **"Independent agent" defined** in Sec. 2 Protocol: "an independent code-verification agent (an automated reviewer, blind to
  results, whose findings were amended into the pre-registration before any result was read)". Appendix A5 repeats it.
- **CORRECTED banners** added (banner only, bodies unchanged) at the top of:
  - `MECHANISM_RESULTS.md`: "a start fixes it completely" and the "Design implication" withdrawn as general statements;
    points to `TORUS_START_RESULTS.md`.
  - `WRAP_EXISTENCE_RESULTS.md`: the "87% -> 18%" drift secondary and "not an attractor" reading withdrawn; points to
    `ACCOUNTS_RESULTS.md` T1.
  - `SIGN_MATCHED_RESULTS.md`: "a replication in a new regime" withdrawn; a learnability result in the rotary slot; points to
    `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md` and `docs/audits/2026-10-12/ring_dial_audit.md` M1; notes the opposition
    score is not wrap-aware.
  CLAUDE.md's sign row was not edited (outside this pass; it already carries the 2026-10-14 correction).
- **Header line 3** kept as a one-line draft marker ("Delete this line before submission"). The "Appendix plan" section is
  gone: the appendices are written.
- **Appendix F discloses DYCK_MDEPTH's four seed-0 smoke runs** (READ item 9).

## Factual correction found during the read

- Draft 4 Sec. 2 said the torus task is scored "with the observation map redrawn for every evaluation walk". It is not:
  `GridWorld` builds one observation map per env seed (`environment.py` l.297-319); each run trains on one map (seed = run
  seed, `train_variant.py` l.523) and is evaluated on one held-out map (env seed 10000, `eval_noise_refine.py` l.114). Only
  the ring redraws the map per walk. Draft 5: "each run trains on one map and is scored on a held-out map (accuracy is
  transfer) ... A 1D ring (... maps redrawn for every walk)". No number changes.

## Placeholders added (no result written)

End of 4.1 and in Limitations, verbatim: "[ROBUST batch pending: recipe robustness on the ring (weight decay 0, lr x3, lr /3)
and a 2-layer d 256 torus contrast; fill only from the registered analysis]".

## Appendices written (`APPENDIX_draft5.md`)

Lettering kept from draft 4's plan; E stays the separate theory file and is cross-referenced by label.
- **A. Models and recipes.** Architecture (from `model.py` etc.), arm-name map, initial-angle arms, optimisation table (from
  `train.train` defaults, the trainers' argparse defaults and the drivers: lr 1e-3, AdamW wd 0.05, 5% warm-up + cosine to 10%,
  torus batch 16 x 1024 x 98 x 900 epochs, ring 128 x 128 x 98 x 300; `--fast-attn` only in GENERALITY / TORUS_START /
  FACTORIAL), evaluation streams, run classes (`stats_core.classify_run`), and a batch table (seeds, trainer, driver,
  independent check, pilot) for every batch cited.
- **B. Floors and gates.** Every floor with the stream it was computed on (the torus-stream issue above).
- **C. Per-seed tables** C1-C9 for every registered contrast in the main text (copied from `*_ANALYSIS.txt`,
  `SIGN_MATCHED.json`, `RING_DIAL_VERDICTS.json`, `GAUGE_RESULTS.md`), C10 SOLVED under four criteria, C11 ring strata
  (Fig. C1), C12 the mechanism's loose ends (moved from 4.3) with the collapse table.
- **D. Constructions.** Per-channel wrap; the balanced construction step by step; exactness and weight scales; why the naive
  one is exact but ill-conditioned (W_out 653-16,320 vs 0.31-0.38); the full |.| arms (M1); continuations and the LOST rule;
  ring vs torus learning rates; what the held model keeps (ACCOUNTS T1).
- **F. Dyck-2 at matched depth** (recipe from `train_dyck.py` / `run_dyck_mdepth.sh`; smoke-run disclosure).
- **G. Navigation in words** (fresh-seed numbers first; AUDIT re-score; CORRECTED clock note).
- **H. Faithful CARoPE and CoPE** (implementations, P6 / P7 pointers, the three CoPE batches, per-seed T4, the post hoc
  diagnosis).
- **I. The toy** (exploratory table from `MECHANISM_PREREG.md` A1 and ACCOUNTS T3).
- **J. The clock side** (draft 4's text plus the registered CLOCK_SIDE accuracy contrasts, all unmeasured).
- **K. Depth** (draft 4's text; floor stream noted).

## Rule-28 read of draft 5 (end to end)

- Cross-references: every appendix letter cited in the main text exists (A, B, C, D, E, H, I, K; F and G are cited from K and
  A5; J from no main-text sentence, it stands as a supplement). Section references 4.1-4.6 and Sec. 1-3 checked. Table 1 and
  Table 2 are cited before they appear; Fig. 1 (4.1) before Fig. 2 (4.3); Fig. C1 and K1 belong to appendices C and K.
- Terms defined before use: MapWM, channel, Theta, signed / |.| / softplus, barrier parameterisation, AO, inverse pair,
  cancels / through zero / by wrapping, SOLVED, p_plus, line (Sec. 2); sigma, P1(a), P3, P4, Lemma 1 (Sec. 3); "the wall"
  (P4) before 4.3; Cohen's kappa named in full; "laps" is plain English (4.5).
- Theory appendix v2 cites "draft 4 Sec. 4.x"; the section numbers and headings are the same in draft 5 (4.2's heading now
  says |.|), so its pointers remain valid.
- Claim map ("Do not claim"): no "cannot represent a map"; no fragility or drift claim (D6 records both withdrawals); no
  weight-decay claim; "the wrap is CoPE's obstacle" stated as not supported; no near-wrap design rule ("not a design rule");
  no CoPE near-wrap rescue; no "only the start"; no double dissociation (J); no "replicates Sarrof / Grazzi"; no "the cost
  follows the relative sign" (4.5 heading is a question; "nothing here suggests the cost needs an inverse pair" is a negated
  reading); the barrier account is said not to explain +1/+2 (abstract, Sec. 1, 4.3, 4.5, Limitations).
- Numbers: every main-text number unchanged from draft 4 except wording; draft 4's numbers were checked against their sources
  in `READ_draft4.md`.

## Could not be sourced, or left open

- Per-seed accuracies for ACCOUNTS T2 (C6) are not printed in `ACCOUNTS_ANALYSIS.txt` / `ACCOUNTS_VERDICTS.json`; C6 gives
  per-seed opposite-sign shares, means and SOLVED counts only.
- Torus n-gram and retrace floors on the batches' own walks: not computed (see above).
- Softplus approximate representability: untested; stated as such.
- Venues of Amid-Warmuth, Orvieto, Woodworth, Grazzi still marked * (not re-checked), as in draft 4.
- `git status` shows `train_gauge.py` modified; this pass did not touch it (no .py / .sh / prereg / runs/ file was edited).

## 2026-10-16: own-stream floors (parent session, post hoc)
- Computed the torus blank / n-gram / retrace floors on SIGN_MATCHED's and GENERALITY's own eval walks
  (`docs/audits/2026-10-16/sign_floors.py`) and the retrace floor on the ring (`ring_floors.py`). Torus retrace 0.843 lies
  above the |.| mean 0.821 (3/8 runs below) and softplus 0.781 (4/8); ring retrace 0.860 at p_plus 0.5 lies below |.|
  (0.95-0.97). Sec. 4.1 now prints both; appendix B's "Which stream" note replaced by the computed values; the gates
  sentence ("each above its floor") qualified: one-layer index RoPE is below the retrace floor on the torus and ring.

## 2026-10-16 15:00: ROBUST filled (`ROBUST_RESULTS.md`, registered)
- Sec. 4.1: new paragraph "Recipe and scale" from the registered analysis (cost holds at wd 0, lr /3, torus 2 layers d 256;
  unmeasured at lr x3; at scale the accuracy gap is small, 6/8 |.| descending, SOLVED fails at loss 0.1 post hoc).
- Abstract: "one recipe per task" replaced by the recipe/scale sentence. Limitations: the "one recipe" scope replaced.
- Appendix C13 added (tables copied from ROBUST_RESULTS.md).
