# Review of draft 5: synthesis (2026-10-16)

Five independent agents, report-only: mock review (`MOCK_REVIEW_draft5.md`, 6/10 weak accept, confidence 4/5), numbers
audit (`NUMBERS_AUDIT.md`, ~760 values, 12 errors), theory check (`THEORY_CHECK.md`, 8 checks re-run byte-identical, 4 WRONG,
2 GAP), reproduction (`REPRO_AUDIT.md`, 11/11 registered analyses byte-identical; 648 checkpoints clean; test_guards 13/13),
prior art (`PRIOR_ART.md`, no prior statement of the learnability cost or Lemma 1; four citations missing).

## What stands
- Every registered verdict reproduces from stored artifacts; torus evals re-run bitwise on GPU.
- No main-text number contradicts its source; no post hoc number is presented as registered; no withdrawn claim is made.
- Novelty: nothing found states (a) the learnability cost of a non-negative rotary step or (c) the relative-sign reflection
  for learnable steps; "to our knowledge" survives, scoped to learnable / content-dependent steps.

## Fixes, ranked
### A. Decide (GPU)
1. **Learning rate as a confound** (mock review W1, its top fix). |.| SOLVED rises with lr (0/8 at lr/3, 1/8 at 1e-3,
   4/8 at 3e-3); the comparison is at the signed arm's best lr. Proposed: pre-registered per-arm lr sweep on the
   observation-step-free ring, signed vs |.| at lr 1e-3 / 3e-3 / 1e-2 (n = 8 fresh seeds; best rate per arm reported),
   48 ring runs, ~3-4 h wall. ROBUST already has 1e-3 and 3e-3 at seeds 1400-1407; adding 1e-2 (16 runs, ~1 h) on those
   seeds gives the sweep in-batch only if registered as an extension -- otherwise a fresh 48-run batch.
   Retitle by the result.

### B. Correct (text; no new data)
2. **Theory errors** (THEORY_CHECK): A0 inhomogeneous term is first-order, not exact; Sec. 3 P4 drops omega and "strict
   local minimum" holds only along s; Lemma 1(ii) must say "no map *function*" (|.| excludes the code (1, 2 - 2 pi) on
   +1/+2); appendix H P/32 -> P/16; P7 precision bound labelled as an estimate. Wording fixes listed in the file.
3. **Softplus is representable, exactly** (THEORY_CHECK item 5): omega' = 2 pi / ln 2 makes the annihilated observations'
   softplus(0) step a full turn; built Pos_r4 from SO_s800 matches logits to 2.8e-11. Replace "untested" (Sec. 3, D4,
   Sec. 6) by the construction; existence under training remains untested.
4. **Numbers** (NUMBERS_AUDIT 1-12): Fig. K1 floor 0.507 -> 0.513 (drawn) / 0.509 (own stream); "every other contrast
   holds at 0.1" -> "in Table C10" (ROBUST torus fails at 0.1); ROBUST loss-0.1 check is a declared secondary, not post hoc;
   CANCEL's index numbers pool pilot seeds and had no independent check -> cite GENERALITY's 0.730 in the main text,
   fresh-seed values in B; A4 re-score sentence false for ACCOUNTS / TORUS_START / GAUGE / ROBUST; ROBUST rows in A1/A3/A5;
   "independent audit after the run" unsourced; GAUGE backtrack share 87.5% -> 89.3% (also in GAUGE_RESULTS.md); appendix H
   CARoPE (0, 4] misstates P6; abstract: "8 of 8" needs "at the default recipe (6 of 8 at a third of the lr)" and "and in
   accuracy at scale"; C1-C12 -> C13.
5. **Framing** (mock W2): abstract, Sec. 1 and contributions 3-4 still read the through-zero route as the cause; +1/+2 and
   the wrap-free solved |.| runs say it is not sufficient. Present the route as what signed steps use, not why |.| fails.
6. **Scoping** (mock): budget-scope the 4.1 / 4.2 headings, Sec. 4.4 "does not transfer" (5/8 runs per arm descending),
   contribution 5; ring claim is "rarely reach an exact map" (|.| at 0.95-0.97); "at a trained scale" -> "within ~5x";
   mark the three unmarked declared secondaries; Sec. 2 "every contrast was pre-registered" too broad; cross-batch torus
   comparisons mix --fast-attn and model classes (say so at "a quarter of the one-layer gap").
7. **CoPE SOLVED fragility** (REPRO 3): 5/16 line CoPE runs score 0.9997-1.000 held out but miss the 0.05 loss cut
   (dropout-on training loss); at an accuracy cut the line reads 5/16 vs 2/16. Say so beside the CoPE line sentence.
8. **ACCOUNTS T1 runs are bit-identical to WRAP_EXISTENCE's** (determinism, not replication; rule 6): do not count them as
   evidence for E2 HELD.
9. **Figures**: Fig. 1 uses the unweighted cancelling share (the measure ACCOUNTS T1 found misleading) -- redraw usage-
   weighted or caption the caveat, and fix the denominator description; Fig. 2 shades the all-negative quadrant as closed
   to |.| (omega's sign is free: only the mixed quadrants are closed); no figure shows the headline contrast (add loss
   curves signed vs |.|, CPU from stored losses).
10. **Citations** (PRIOR_ART, THEORY_CHECK): add Shakerinava et al. ICLR 2026 ("optimization, not expressivity" for
    input-dependent rotations; must cite), Helfrich et al. 2018 scoRNN (rotation orientation not learnable, fixed at
    init), Wang and Wang 2026 / Tran et al. 2026 (fixed-frequency RoPE symmetries have no reflection: scopes Lemma 1);
    credit MapFormer v4 Table 2 (CoPE fails navigation) and Figs. 3d / 9b (antipodal action latents); theorem numbers for
    Grazzi (Thms. 1-3, App. C.1), Sarrof (Thm. 2), Amid and Warmuth (Prop. 2); fix the Grazzi and Khavari sentences.

### C. Venue
11. **Length**: ~3,500 words + 2 tables + 2 figures is ~6-7 pages against a 4-page short track. Cut to 4.1-4.3 + Lemma 1
    for a 4-page track, or target an 8-page track (CCN 2027 proceedings) as is.
12. Optional (mock W5): LRU's exp(.) phase is the barrier parameterisation actually used in the literature; untested.

## Housekeeping done with this synthesis
Run-dir eval / construct / md5 JSONs of ACCOUNTS, TORUS_START, GAUGE and ROBUST (and the two pilots) committed, as earlier
batches did (REPRO note 1); review reports and downloaded paper texts committed (PDFs not).
