# End-to-end read of draft 4 (2026-10-15, after the GAUGE fill)

Draft `docs/paper/DRAFT4_workshop.md`; appendix `docs/paper/APPENDIX_theory_v2.md`. Rule 28: read start to finish as a
reviewer, every main-text number checked against its results file (CORRECTED / AUDIT blocks first) and the 2026-10-15
re-analysis outputs. Edits are logged old -> new with sources in `DRAFT4_changes.md`, section "2026-10-15 end-to-end read".
Nothing committed.

## (a) Verdict

**Ready to share for co-author reading after the edits below (made). Not ready to submit:** the main text is ~4,300 words
against a ~3,500 target (cuts in (c)1), and appendices A-D and F-I are still plans. No number in the main text contradicts
its source; no registered label is stated more strongly than its results file. After GAUGE no sentence now says or implies
that the cost follows the relative sign, or that the barrier-at-zero account explains every failure. Two places did imply
that, and both are fixed (D5, D10). The title is the remaining tension (judgement call 2).

## (b) Edits made (17 in the draft, 7 in the appendix; details and sources in `DRAFT4_changes.md`)

Draft:
- **Post-GAUGE consistency.** Lemma 1(ii) no longer says "no cost is predicted". It now says the barrier excludes no
  sign-coherent code, and that the learnability corollary is a prediction which Sec. 4.5 did not confirm (D5). The 4.3
  conclusion now lists "or neither" among the failure accounts (D10). Sec. 1 now says the paper does not explain the +1/+2
  difficulty (D1).
- **4.5 labelling.** G3 added. The laps comparison is labelled a descriptive post hoc reading of a declared secondary
  (D13). "A third ring batch" is now specific (D12). The file name is gone from the heading (D11).
- **Wrong statements.** "Past the wrap" -> "past pi" in contribution 4 and in Related work (D2, D15). CoPE "can represent
  them" -> "the ring map" (D3). Signed collapse "10-17%" -> "10-16%" (D9). CoPE "1 of 32" is now scoped to the paper's
  initialisation (D14). Appendix K "at or below the floor" -> "near or below" (D16).
- **Notation.**
  - s was used for three things (the pre-|.| step, the key position, Howard's rate variable). It now means only the
    pre-|.| step (D4).
  - q~ and z are defined (D4, D7).
  - "Non-blind" is dropped from the main text, since it was defined only in the appendix (D6).
- **Tables.** The draft had a Table 2 but no Table 1. Now 4.3's table is Table 1, and the GAUGE table is captioned
  Table 2 (D8).

Appendix:
- The P4 and Remark R text still said "not established / not done" for checks the 2026-10-15 re-analyses had since run.
  It now reports their results: the wall gradient reads MIXED, and the flow does not predict cancellation channel by
  channel (kappa +0.09) (A4-A6).
- T1(ii)'s heading now records the GAUGE result, and a wrong in-section cross-reference is fixed (A2, A3).

## (c) Open judgement calls, most important first

1. **Length: cut ~800 words to reach ~3,500.** Proposed cuts, about -790 in total:
   - 4.3 "What fits loosely" (~400 words): keep two sentences (rescued |.| models cancel in only 23% of channels and the
     one-channel flow does not predict which; collapse at the wall is common but its gradient check is mixed). Move the
     rest to appendix C/E. About -300.
   - Related work (423): drop the GridPE / TEM sentences, keeping one clause on TEM-t / MapFormer. Drop the repeat of
     Sarrof / Grazzi / Mamba-3 parity, which Sec. 1 already gives. Make "Depth" one sentence pointing to appendix K.
     About -170.
   - Abstract (239 -> ~170): drop "(0-2 wherever a registered contrast fires)" and the "(7/8; 1/8 with them frozen)"
     numbers, and merge the last two sentences. About -70.
   - 4.1: fold the threshold-sensitivity sentences into one ("at loss < 0.1 p_plus 0.5 no longer fires, 4/8 vs 8/8,
     p 0.077; every other contrast holds under four criteria, appendix C"). About -50.
   - Sec. 3, collective route: drop the flow numbers, which 4.3 repeats. About -50.
   - Sec. 2, Protocol: move the appendix-only caveats (navigation in words, Dyck) to the appendix. About -50.
   - 4.6: compress the post hoc line-threshold sentence to a clause. About -40.
   - Limitations: drop the protocol sentence that repeats Sec. 2. About -30.
2. **The title after GAUGE.** "A barrier at zero" names an account that the paper's own registered test (4.5) could not
   confirm, and |.| also struggles on +1/+2, where nothing has to cross zero. Proposal: lead with the measured fact. For
   example, "Rotary phase steps parameterised to be non-negative can represent a map but rarely learn one (one-layer
   models, synthetic maps)", with the barrier kept in the abstract as the route that is removed. The alternative is to
   keep the title and add one abstract clause saying the barrier does not account for the +1/+2 case.
3. **Softplus and "can represent".** P1(a) and the M1 construction are for |.|. A softplus step cannot be exactly zero,
   so the observation steps cannot be annihilated, and nothing in the draft or appendix shows that the softplus arms can
   represent the map. The title ("parameterised to be non-negative can represent") covers softplus. Proposal: scope the
   representability sentences to |.| ("the |.| class can represent ...") and say softplus is untested for exact
   representation. A one-paragraph appendix remark could show approximate representation (large negative
   pre-activations), but only if it is checked first.
4. **Torus floors missing in 4.1.** Sec. 2 promises floors beside accuracies, but 4.1's 0.821 vs 0.998 has none. Proposal:
   add "(floors: blank 0.507, n-gram 0.576)", after confirming that `SIGN_MATCHED`'s eval stream is the T = 1024 stream
   those floors were computed on. `SIGN_MATCHED_RESULTS.md` reports no floor; the numbers come from the GENERALITY /
   TORUS_START / FACTORIAL stream.
5. **Source files out of step with the draft (reviewer-checkable).** Three results files still say things the paper has
   dropped, and none carries a banner:
   - `MECHANISM_RESULTS.md`: "Design implication" and "a start fixes it completely" (withdrawn by TORUS_START).
   - `WRAP_EXISTENCE_RESULTS.md`: "drifts 87% -> 18%" (withdrawn by ACCOUNTS T1).
   - `SIGN_MATCHED_RESULTS.md`: "replication in a new regime" (corrected 2026-10-12/14).

   Proposal: add CORRECTED banners before the repo or the files are shared. This is outside this pass's three files.
6. **Contribution 3 is now half negative.** It promises "what the barrier removes", and then reports a test that "cannot
   read" the case it was designed for. Proposal: merge it into a short statement, "Lemma 1 (relative-sign gauge) says
   which codes the barrier excludes; a registered test of its learnability corollary was inconclusive (Sec. 4.5)". Or move
   Lemma 1 into contribution 2 as the representability side.
7. **"Independent agent" in Protocol.** A reader may not know what this means. Proposal: "an independent code review,
   blind to results (an automated agent)". Keep it honest about what it was, and define it once.
8. **Repeated material beyond item 1.** The Howard framing appears in Sec. 3 and in Related work. The protocol caveat
   appears in Sec. 2 and in Limitations. The 28% flow number appears in Sec. 3 and in 4.3. Removing these repeats is part
   of the cut in item 1.
9. **Header line 3** ("Draft 4 ... Delete this line before submission") and the "Appendix plan" section must go or be
   written out before submission. Appendix F should disclose DYCK_MDEPTH's four seed-0 smoke runs (`DRAFT4_changes.md`
   Sec. 4).

## (d) Main-text length

Counted with `wc -w` from "## Abstract" to the line before "## References", including tables and formula tokens:

| | before | after |
|---|---|---|
| words | 4,157 | 4,302 |

The +145 is the clarifications in (b). By section after the edits: abstract ~240, Sec. 1 ~375, Sec. 2 ~385, Sec. 3 ~715,
4.1 ~275, 4.2 ~230, 4.3 ~780, 4.4 ~105, 4.5 ~325, 4.6 ~220, Sec. 5 ~425, Sec. 6 ~230. At ~4,300 words the draft is over a
4-6 page workshop limit. The cuts in (c)1 bring it to ~3,500.

## (e) The three weakest points a reviewer would attack

1. **Generality: one layer, synthetic maps, one recipe per task, two barrier forms.**
   - Learning rate, weight decay from scratch, optimiser and depth were never varied for the sign contrast. The mock
     review's E1 was not run.
   - Softplus is confounded by its initialisation.
   - On the torus, latent rank slack is uncontrolled.
   - There is no exp(.) (LRU) or recurrent baseline.

   *Conceded:* yes. The title, the abstract ("within our budgets ... one recipe per task") and the Limitations list each
   gap. The text cannot answer the attack; an experiment can (E1).
2. **The mechanism does not explain the failure.** The paper is framed around a barrier at zero, yet:
   - its own registered test (4.5) finds |.| equally poor on +1/+2, where no zero crossing is needed;
   - P3 (unreachable wrap) and P4 (wall) are not separated, and the wall's gradient check is mixed;
   - the one-channel flow does not predict which channels cancel;
   - the start lever is ring-only.

   What is established is narrower: on the ring, signed steps have a collective route through zero, and the barrier
   removes it. *Conceded:* in the body, yes, explicitly (Sec. 1 D1, 4.3 closing, 4.5, Limitations). Only partly in the
   framing: the title and contribution 3 still lead with the barrier (judgement calls 2 and 6).
3. **The ring evidence is a loss threshold with a small accuracy gap.**
   - On the ring, |.| reaches 0.95-0.97, a gap of -0.03.
   - At p_plus 0.5 the SOLVED contrast does not survive a loss threshold of 0.1 (4/8 vs 8/8, p 0.077, post hoc).
   - The accuracy evidence rests on the torus: -0.177 in a batch with no independent pre-launch check, and -0.120 /
     -0.104 in the checked observation-step-free batches.

   *Conceded:* yes (4.1, Limitations, Sec. 2 Protocol). A reviewer may still say the ring is a convergence-speed result.
   Answer: RING_BUDGET (3x training, 3/8 -> 3/8) is in 4.1, and the dichotomy holds at three other criteria (appendix C).

A fourth point is close behind: **novelty.** Lemma 1 is a one-line identity, and the representational point is Grazzi /
Mamba-3's. *Conceded:* yes ("the identity is elementary"; Related work claims "their conjunction and the measurements, not
any single ingredient").
