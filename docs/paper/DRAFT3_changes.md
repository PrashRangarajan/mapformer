# Draft 3 vs draft 2: changes and number sources (2026-10-15)

Draft 3: `docs/paper/DRAFT3_workshop.md`. Draft 2 (`docs/paper/DRAFT_workshop.md`) is unchanged. Inputs: advisor plan
`docs/plans/2026-10-14_next_steps.md`; claim map `docs/paper/CLAIM_MAP_clock_map.md` (incl. "Do not claim"); theory
appendix `docs/paper/APPENDIX_theory.md`; lit note `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`. No GPU, no new experiment.
Main text ~3,650 words by `wc -w` from Abstract to Limitations (tables and formula tokens included), plus ~230 in captions.

## Structural changes

| # | change | reason / source |
|---|---|---|
| 1 | New title: "Sign in rotary position is a matter of learnability: non-negative phase steps can represent a map but, within our budgets, rarely learn one" | old title ("...does not learn one") contradicted by MECHANISM M2 (8/8 from past pi); plan A3 |
| 2 | Abstract rewritten: representability attributed to Grazzi et al. / Mamba-3; claim = learnability + mechanism + regime; T1 stated; ring-only start lever; budget scope | plan A1, A3; lit note Sec. 1 |
| 3 | Intro: representational sign theorems (Sarrof, Grazzi) do not cover rotations; "same advice, different reason" | lit note Sec. 1 item 4 |
| 4 | Results reordered: 4.1 sign cost, 4.2 existence, 4.3 mechanism (+T2), 4.4 torus start (new), 4.5 published methods, 4.6 depth (was 4.1), 4.7 clock side (was 4.6, natural text cut) | plan D6; task item 2 |
| 5 | Sec. 2 rewritten as "Background and theory": A0, P1(a) scoped with prior-art attribution, Theorem 1 + corollaries (i)-(iii) (new), P3-P5 briefly (P4 cited to Woodworth / Amid-Warmuth), Remark R as an account | APPENDIX_theory.md; plan T-1, T-2 |
| 6 | Old Prop. 1 sentence "a generic signed code has no exact rank-r non-negative counterpart (sketch)" removed; replaced by "proved for r = 1, conjectured for r >= 2; appendix E" | APPENDIX_theory.md P1(c), P1(e); plan A cut list |
| 7 | Old Prop. 2 (rescaling) moved from main text to appendix E as mathematics only; its use as an explanation of drift and the torus knock-out removed | ACCOUNTS T1; claim map "Do not claim" |
| 8 | Drift sentence "87% -> 18% of channels; torus 100% -> 69%" deleted; replaced by usage-weighted 0.99 -> 0.98 | ACCOUNTS T1 |
| 9 | "Design implication: initialise a Case-II rotary step near the wrap" deleted; Sec. 4.4 states the start lever is ring-only | TORUS_START TS1; claim map |
| 10 | New Sec. 4.3 paragraph: collective escape (frozen latents 1/8 vs 7/8) | ACCOUNTS T2 |
| 11 | CoPE: lattice-diagnosis paragraph (4.5 in draft 2) moved to appendix H; main text says learnability, obstacle unidentified, precision a plausible account; line result added | ACCOUNTS T4; COPE_DIAGNOSIS (post hoc) |
| 12 | Depth section replaced by the lit note's corrected text (Liu Lemma 6 + 8, not Theorem 3; not a capacity claim; "to verify" removed; precision-vs-short-gap sentence dropped) | lit note Sec. 2 |
| 13 | Related work replaced by the lit note's Sec. 4 (condensed); LRU contrast sentence rescoped (ring past wrap helped; torus no start helped) | lit note Sec. 4; TORUS_START |
| 14 | Natural-text text (TinyStories, code) cut; one clause in Limitations | plan A "Cut" |
| 15 | Clock side kept as one paragraph (two registered UNMEASURED); block-vs-delay given as an untested account | CLOCK_SPEED(_REP); THEORY_REVIEW Sec. 6 |
| 16 | Ring strata readout added (post hoc) to answer "SOLVED is only a loss threshold" | FIGS_NOTES ring strata; plan A6 |
| 17 | Figures renumbered by order of appearance: Fig. 1 = wrap share (was Fig. 2), Fig. 2 = mechanism (was Fig. 3, now made), Fig. 3 = factorial strata (was Fig. 1); Fig. C1 = ring strata (appendix) | restructure |
| 18 | Toy reduced to three lines in appendix I | plan A "Cut"; ACCOUNTS T3 |
| 19 | Table 1: Mamba-3 row notes sign in theta_t; GRAPE-AP in the real-s row; softplus arm no longer called GRAPE-AP | THEORY_REVIEW Sec. 1; APPENDIX A0 |
| 20 | "Our sign cost replicates theirs" removed everywhere | lit note Sec. 1 item 4 |

## Number sources (every number in the main text)

| number(s) in draft 3 | file (registered unless noted) |
|---|---|
| torus Abs - Signed -0.177 (0.821 vs 0.998), p 0.0002, 0/8 vs 8/8; softplus -0.218, 1/8 | `SIGN_MATCHED_RESULTS.md` |
| ring 2/8, 1/8 vs 8/8 (Fisher 0.007, 0.0014); p_plus 0.9 4/8 unmeasured; acc 0.973 / 0.968; floor 0.554; index 0.717 | `RING_DIAL_RESULTS.md` |
| ring index 0.730 | `GENERALITY_RESULTS.md` |
| 3x budget 3/8 vs 3/8 | `RING_BUDGET_RESULTS.md` |
| cancelling shares 0.79 / 0.78 / 0.64 / 0.98 signed; 0.06 / 0.03 / 0.06 / 0.11 |.|; 0.03 softplus | `docs/paper/figs/fig_wrap_share_out.txt` (post hoc) |
| observation-step-free 0/8 vs 8/8: ring (WRAP_EXISTENCE E3, MECHANISM M4); torus 0/8 vs 8/8 (GENERALITY E3), 1/8 vs 8/8 (TORUS_START TS4) | those files |
| ring strata 0.96-0.99 -> 0.87-0.96; 8/8, p 0.0078; gap-matched -0.063 / -0.007 / -0.032 | `docs/paper/figs/FIGS_NOTES.md` (post hoc) |
| angle error <= 1.4e-6 (torus), 1.2e-6 (ring); ~5x weights (ring); latents cos -0.987..-0.998; ~10 epochs; ring held 8/8, 0/8 LOST | `WRAP_EXISTENCE_RESULTS.md`, `GENERALITY_RESULTS.md` |
| torus continuation peak loss 0.11-0.21 (0.110-0.208) vs 0.004-0.017; 6/8 LOST; 8/8 SOLVED 1.000; lr 1e-3 ring / 1e-4 torus | `GENERALITY_RESULTS.md`, `GENERALITY_ANALYSIS.txt` |
| usage-weighted 0.99 -> 0.98 | `ACCOUNTS_RESULTS.md` T1 |
| Table 2; M5 -0.092 [-0.135, -0.045], Fisher 0.0014; M2 +0.109 [+0.085, +0.135], p 0.0002; M2b Fisher 0.0070; M6 8/8 vs 8/8; mobility 13-22x / ~5x | `MECHANISM_RESULTS.md`, `MECHANISM_ANALYSIS.txt` |
| 98-99% opposite-signed, 0% wrapped; 88% / 100% by wrapping | `docs/audits/2026-10-14/mech_escape_out.txt` (post hoc) |
| T2: 1/8 vs 7/8, Fisher 0.010 (0.0101), -0.075 [-0.130, -0.020]; 15% vs 82%; F2 7/8 vs 8/8 | `ACCOUNTS_RESULTS.md`, `ACCOUNTS_ANALYSIS.txt` |
| Fig. 2 caption shares 0.06 -> 0.89, 0.04, 0.05, 0.07 -> 0.23 | `docs/paper/figs/fig_mechanism_out.txt` (post hoc) |
| torus start TS1 0/8 vs 1/8, +0.005 [-0.060, +0.070]; SO4_high 2/8; 5/8 descending; TS4 1/8 vs 8/8, Fisher 0.0014 | `TORUS_START_RESULTS.md`, `TORUS_START_ANALYSIS.txt` |
| CARoPE 0.667 vs 0.722 (p 0.040); ring 0.741 vs 0.730; CoPE 0.887 0/8, p 0.0002, +0.157 | `GENERALITY_RESULTS.md` |
| near-wrap CoPE table start unmeasured | `WRAP_COPE_RESULTS.md` |
| CoPE line 1/16 vs ring 0/16; signed line 8/8; index line 0.772 | `ACCOUNTS_RESULTS.md` T4 |
| factorial 1.000 / 0.719 / 0.860 / 0.864, p 0.029, floors 0.507 / 0.576; strata 0.74-0.79 -> 0.90-0.96, 0.37-0.53 | `FACTORIAL_RESULTS.md` |
| words 0.969 vs 0.505, p 0.0002, 7/8 vs 0/8; re-score 0.993 (appendix G) | `TEXTWORLD_RESULTS.md` (AUDIT block) |
| Liu: Lemmas 6 and 8; C_4 median < 90% at 1-15 layers (max 85.8) | lit note Sec. 2; checked against `papers/new_2026-10-14/2210.10749/liu2023.txt:1705-1751` (Fig. 9, medians) |
| clock: -0.197 (p 0.053, n 16); -0.106 (p 0.19, n 16) | `CLOCK_SPEED_RESULTS.md`, `CLOCK_SPEED_REP_RESULTS.md` |
| Remark R: ~0.5 rad capture, 12% cross zero | `APPENDIX_theory.md` Remark R; `THEORY_REVIEW.md:178-179` |
| Theorem 1 bitwise checks; P1(a); P3 distances; P6 (0, 4] | `APPENDIX_theory.md`, `docs/paper/appendix_checks/*_out.txt` |
| Dyck +0.353 / +0.130 / +0.045 / +0.024 (appendix F) | `DYCK_MDEPTH_RESULTS.md:61-64` |
| P2 frozen probe 0.80 vs 0.998 (appendix E) | `WRAP_EXISTENCE_RESULTS.md` secondary |

## Not verified first-hand in this pass
- Liu et al. Lemma 6 / 8 content and the GridPE, Khavari, Hahn-Rofin, Cornford citations: taken from the lit note
  (Khavari and GridPE abstract-level there; several author lists "from memory, verify").
- The "~5x" weight-scale statement is stated for the ring only; no torus figure found, so the draft scopes it to the ring.

## Do-not-claim check (claim map, binding)
Representability of non-negative rotations is attributed to Grazzi / Mamba-3, never claimed. No "cannot represent a map",
no "fragile under training" (P2's frozen probe only, appendix), no drift claim, no weight-decay claim, no "wrap is CoPE's
obstacle" (stated as refuted), no general near-wrap design rule (stated as ring-only), no near-wrap CoPE rescue, no "only
the start, not the constraint", no double dissociation, no "replicates Sarrof / Grazzi".
