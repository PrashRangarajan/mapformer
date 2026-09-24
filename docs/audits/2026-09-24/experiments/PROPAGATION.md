# PROPAGATION CHECK: retracted or superseded results still presented as findings

Read-only audit, 2026-09-24. Retraction IDs R1-R40 are from `RETRACTIONS_LIST.md`. Line numbers
are from `grep -n` / `awk NR` on the working tree. For every target file the working tree matches
HEAD; the only untracked files are the RANK_PROJ/rank_matched_e900 ones.

Status codes:
- **(i)** presented as a current finding, with no correction in the document;
- **(ii)** a correction exists in the same document but not beside the claim, or the claim is qualified;
- **(iii)** inside a correction block. These are not listed, except where a document handles an item well.

"stale" means the document was last edited before the retraction. "POST" means the document was
edited after the retraction and still carries the claim, which is the more serious case.

## 0. Timeline

| document | last substantive edit | notes |
|---|---|---|
| positional_review.tex | 2026-09-07 22:58 | Predates AUDIT_2026-09-10, COMP_HEADROOM/HIER_RECHECK (09-08), PAPER2X2 and MONOTONE (09-13), RANK_MATCHED (09-23). **Postdates** REFINE (08-31), DXR (09-05), LOCALISATION and ACCUMULATOR (09-06), FLIPFLOP (09-07 13:47), MAPPOPE_R4 (09-07 22:51). |
| axes_measured.tex | 2026-09-08 01:00; 09-11 edits touched only the WM-additive and phase-spread sentences | The 09-11 commit ("corrections three audits forced") did **not** apply audit #2 (recency does not need a clock). Predates PAPER2X2 and COMP_HEADROOM. |
| mapformer_math.tex | 2026-09-07 22:17 | Same window as the review. |
| RESULTS_INDEX.md, README.md | 2026-09-11 | Predate PAPER2X2. |
| BASELINE_TABLE.md | 2026-09-07 | |
| EM_WM_STATE.md | 2026-09-12 | |
| report/STORY.md | 2026-09-13 20:56 | About 2.5 h before PAPER2X2_RESULTS.md (23:19). |
| report/report*.tex | 2026-09-20 | |
| report/language_summary.html | 2026-09-23 13:24 | The live shared report. Postdates everything except RANK_MATCHED e900 (09-24). |
| .claude-memory/project_state.md | 2026-09-23 19:47 | Byte-identical to `~/.claude/.../memory/project_state.md`. |

## 1. Additional retractions found during the check (not in RETRACTIONS_LIST)

- **R40b. "PoPE is inert with an index phase / both index arms on the floor".** This is a consequence
  of the 16-epoch recipe. Under the converged PAPER2X2 recipe, PoPE-Flat scores 0.679 and RoPE 0.805
  at T=128. The encoding main effect is **detectable** at every length: -0.049 at T=128, and
  +0.114 / +0.189 at T=512 / 1024. So "encoding +0.003, does not move it" is also superseded.
- **R41. The loop-floor claim.** "The loop arm never fails, 8/8 seeds >= 0.77; the loop raises the
  floor" was RETRACTED in REFINE_RESULTS.md on 08-31: pooled over 16 draws it is 0.803 +/- 0.200 with
  1/16 failures. Same-seed retrains on Match-Query drift 0.185 per seed, so the **paired**
  estimates are unreliable: loop +0.414 / +0.348, interaction +0.315, and MQ_RANK's +0.154 / +0.149.
  report.tex handles this; report/VERIFY.md E2/E3 flagged STORY.md.
- **R42. Audit 2026-09-10 #2 scopes Thm 1.** "Mutually exclusive by construction" holds only for a
  scalar accumulator. Recency does **not** need a clock: a signed rewind solves it exactly
  (1423/1423; frozen install 1.000). So calling recency "a task built to want a clock" or "a task
  whose answer depends on elapsed count", and saying "the ordering inverts", is withdrawn. The
  crossover data stand. MONOTONE (09-13) adds that the recency cost is small but detectable for
  SRoPE (-0.068) and for EM (-0.198).
- **R43. The sep-vs-P0 "no additive fallback / outvoted" initialisation-pathology account.** It is
  withdrawn in EM_WM_STATE.md:272-274 and AUDIT #1/#3.
- **X-FF. The review recommends Flip-Flop LM as "the published dataset that would settle it".** It
  was run before the review's last edit and cannot discriminate, because it only asks k=1
  (axes_measured.tex:618-658).

## 2. Hits by document

### report/language_summary.html (the live shared report, 09-23)

| line | phrase | R# | status |
|---|---|---|---|
| 281 | "which is also measured at the training length: path integration beats a matched index model by **+0.461** (n=8), reaching 0.99 where the index model **sits on a 0.506 floor**" | R40 (item a) | **(i) POST.** This number comes from the 16-epoch LinearLR recipe. Converged, it is +0.243 with RoPE at 0.805. report.tex:626 itself says "Index arms are undertrained under this recipe". The page's "matched vs mismatched length" thesis (221, 384) leans on it. |
| 234 | "On text, code and music, **plain PoPE still wins**." | related to R4/ABLATE | **(i) overclaim.** The page's own table (353) has code as "PoPE ~ RoPE". ABLATE shows PoPE - RoPE on code is +0.0034 (worse). CODE_DECAY shows PoPE-Decay detectably worse than RoPE-Decay. On enwik8 the repo has MapPoPE numerically *ahead* of PoPE (1.3740 vs 1.3746, n=1; axes_measured.tex:727 has 1.3786 vs 1.3806). enwik8 is listed under Tasks (223) but no enwik8 result appears on the page. |
| 351-361 vs 408 | Ordering by "kind of position: signed / a clock"; "Adding path integration to PoPE is not [good], on clock-like tasks", against "Refuted: ... a clock-versus-map theory of when each design helps" | R6 | **(ii) internal tension.** The page withdraws a clock/map theory, then organises its recommendation by it. The sign-to-map/clock dichotomy stands; the cross-task PoPE corollary does not. The wording does not separate the two. |
| 236, 313 | RoPE + envelope is "the best of all eight code models tested" | none (caveat) | Unverified superlative. The comparison crosses batches (runs/code vs runs/code_decay). It uses the 40-window best_val_bpc readout that CODE_DECAY itself calls min-biased. RoPE-Decay leads MapWM-Decay by 0.0033, against a batch floor of 0.0021/0.0028. |

Handled correctly: code OOD (407), depth substitution (409), F1 (370-381, Hewitt used in the tables),
non-negativity (408), Indirect Indexing (290, no 0.965 claim).

### RESULTS_INDEX.md (09-11)

| line | phrase | R# | status |
|---|---|---|---|
| 22-38 | THE HEADLINE: "matched recipe ... RoPE 0.514 / 0.989, PoPE 0.509 / 1.000 ... **Both index cells sit at the floor** ... Position ~0.46, the encoding ~0.003" | R40, R40b | **(i) stale.** The table is the **n=3** INDEX_BASELINE_PAPER_TASK.md table, not n=8, and "matched recipe" means the 16-epoch LinearLR recipe that rule 10 in the same file condemns. |
| 147-152 | "**Use r=4, not the paper's r=2.** +0.085 at T=1024 (8/8, t=3.57)" | item (b), R23 | **(i).** It does not say T=1024 is 8x the training length, and it gives the recommendation generally. |
| 154-159 | "MapPoPE-Flat is the strongest configuration ... **PoPE is inert without path integration (0.509)** ... r=4 has never been applied to it; a batch is running" | R40b, R23 | **(i) stale.** MAPPOPE_R4 (09-07) found +0.019 (unmeasured), i.e. MapWM-family only. The note is four days stale at the 09-11 edit. |
| 167-170 | "The loop composes ... `r=4 + loop x4` 0.986, 8/8 >= 0.941 ... positive interaction +0.149" | R41 | **(i) POST.** This is a paired estimate on non-reproducible pairs, retracted by REFINE (08-31). |
| 197-198 | "**Hierarchy** helps compositional transfer and long-horizon aggregation" | R30, R38 | **(i) POST.** HIER_RECHECK (09-08) says this is unmeasured, and the task is recipe-limited. |
| 172-178 | Separate q0/k0 "+0.358 against it ... four tasks" | R26 | (ii). The recency reversal is noted beside it. |
| 387-388 | "the +24.8pp result has never been through rule 2" | R11 | (ii). The interpretation is withdrawn at 317-322. |
| 389-391, 427 | DOG_RESULTS.md is flagged as vacuous but still listed under "other current" | R33 | (ii) |
| 399 | "Vanilla (WM) on the family tree ... no plain WM" | none | Stale open item: BASELINE_TABLE F closed it. |

### BASELINE_TABLE.md (09-07)

| line | phrase | R# | status |
|---|---|---|---|
| 25, 37-39 | torus "+0.003 (unmeasured) / **+0.461**"; "the encoding does not move it measurably" | R40, R40b | **(i) stale.** The provenance line (59) does say "16 epochs". Line 336 of the same file calls the 16-epoch budget "a budget artifact" for Level15. |
| 132-138 | "**Cite the hierarchy gap as 0.415 vs 0.285 (+0.130)**" | R38, R30 | **(i) stale** (COMP_HEADROOM and HIER_RECHECK are 09-08) |
| 101, 111 | Match-Query 0.888, corrected in place to "cite 0.730 (n=5)" | R18 | (ii) |
| 346 | "the +24.8pp headline has no ExtraHead arm" | R11 | (ii). Withdrawn at 340. |
| 297 | Knob-sweep baseline +0.438 | R40 scope | Note: also 16-epoch, with the index arm at the floor. The same caveat applies to every knob-sweep effect, including +0.050 and +0.488. |

### EM_WM_STATE.md (09-12)

| line | phrase | R# | status |
|---|---|---|---|
| 285-286 | Sec 5, "Current best account": "**On every map task they tie to within 0.004**" | R7 (tie clause) | **(ii).** It is struck at 117-120 in the same file, but restated in the summary section. |
| 189 | "why training never finds the solution" | R8 | (ii). Corrected only in the top block. |
| 217 | Rewind-probe row: "**no arm learns a rewind**" | R8 | (ii). The table row has no inline correction; only the top block (15-24) has one. |
| 258 | The withdrawn-list reason cites "No from-scratch arm learns one", which is itself the withdrawn probe | R8 | (ii) |

### README.md (09-11)

| line | phrase | R# | status |
|---|---|---|---|
| 268, 279-280 | "Level 1.5 InEKF adds a **bounded-error correction**"; "**The wrap on innovation is what keeps θ̂ bounded at OOD sequence length — the key mechanism for length extrapolation.**" | R9, R31 | **(i) POST.** ACCUMULATOR (09-06): range(θ̂) 285.6 vs range(θ_path) 283.9. |
| 249-250 | GSF_NoDrop "**Recommended for landmarks / sparse-cue tasks**" | R11 (lm200) | **(i).** Only the file-top banner (3-27) contradicts it. |
| 376-381 | "Honest framing ... two architectural improvements (NoDrop, GSF) that match or beat TEMFaithful ... on lm200" | R11 | (ii). Contradicted by the banner and line 60. |
| 132-147 | "Cognitive-map necessity ... MapFormer family with correction wins all of them" | R11 | (ii). Line 60 voids rows 1, 2 and 5. |
| 115, 301 | "App. A.4" | none | Minor. The 09-11 commit corrected the citation to App. A.7 elsewhere, but not here. |

### report/STORY.md (09-13, before PAPER2X2)

| line | phrase | R# | status |
|---|---|---|---|
| 423 | "the loop's contribution is mostly to the floor (**8/8 seeds >= 0.77**)" | R41 | **(i) POST** (REFINE 08-31). report/VERIFY.md E2 already flags it. |
| 369-373 | Paired "+0.414 / interaction +0.315"; C13 "+0.149" | R41 | (i) |
| 146-148, 860-861 | "Encoding +0.003 ... a POWERED NEGATIVE"; "Negligible on the paper task" | R40b | **(i) stale.** The recipe weakness is flagged at 50-51 and 836. |
| 839-841 | "the torus separation at a converged recipe is an OOD-length effect" | R40 | **(i) stale.** PAPER2X2 has +0.243 raw at T=128, 8/8. |
| 22, 31 | r=4 +0.085 at T=1024 | (b) | (ii). Line 31 says "every accuracy effect lives at OOD length". |

### positional_review.tex ("the presentable one", 09-07)

| line | phrase | R# | status |
|---|---|---|---|
| 378-389 | "**Two thresholds bracket the useful range.** Below, a packing argument ... N^{-max(D/ρ,1)}" | R22 | **(i) POST** (DXR 09-05: r=2 is unimpaired at D=5). axes_measured.tex:356-357 says "the packing account of the companion review does not explain it". |
| 1007-1018 | "**The critical dimension** ... worth importing ... This is a mechanism for length-extrapolation failure" | R28 | **(ii) POST.** It is refuted at 1170-1175, about 160 lines later, and presented unqualified here. CLAUDE.md says it was "removed from both documents". |
| 830-839 | Single p0 beats separate by "+0.358 ... because with **no additive fallback** a mis-set position factor multiplies the whole score instead of being **outvoted**" | R7, R43, R26 | **(i) stale.** axes_measured struck its own WM-additive text on 09-11; this passage in the review was not touched. |
| 43-47, 86, 145-151, 511, 526 | "**mutually exclusive by construction**" | R42 | **(i) stale.** The audit scoped this to scalar accumulators. |
| 148-150, 573-580, 1115-1119 | "on a task **built to want a clock** ... the ordering inverts as predicted"; "a task whose answer depends on **elapsed count**" | R42 | **(i) stale** |
| 1072 | "which is why hierarchy loses here and **wins on compositional transfer**" | R30, R38 | **(i) stale** |
| 1067, 1137-1138 | "r=2 degrades"; "**r=2 is the worst setting and r=4 fixes it for 384 parameters**" | (b), R23 | (i). It has neither an extrapolation qualifier nor the MapWM-only scope. |
| 1193-1194 | "The published dataset that would settle it is **Flip-Flop LM**" | X-FF | **(i) POST** |
| 101-117 | "Four surveys" then "none of the five"; cites "Zhu et al." | R24 (handled) | Minor. The count is inconsistent, and the survey's first author is Jiaheng Liu (Zhu is co-first). CLAUDE.md says "Zhang". |

Handled correctly: R9 (1068, 1141-1146), R24 (107-117), R27 (1154-1168). The review does **not**
cite +0.461.

### axes_measured.tex (results paper, 09-08 / 09-11)

| line | phrase | R# | status |
|---|---|---|---|
| 46-47, 139, 147 | Abstract: "a parameter-matched index code **sits on the 0.506 blank floor**". Transfer table: torus 0.994 / 0.987 / 0.967 vs 0.530 / 0.509. Text: "An index code ... cannot locate itself". | R40 | **(i) stale.** Provenance error too: the Setup paragraph (193-195) says every run uses "300 epochs ... cosine ... lr 1e-3" unless stated, but these are 16-epoch LinearLR numbers. |
| 418-423 | "position main effect is **+0.461** (8/8, **MDE 0.027**) and the encoding main effect is +0.003" | R40, R40b | **(i) stale.** Also inconsistent with report.tex:626, which says "No sd or MDE was recorded for the position effect". |
| 445, 448 | Anchors table: "+0.461 torus"; PoPE "**both arms on the floor without it**" | R40, R40b | (i) |
| 37-38, 525-528, 544, 548 | "on a task **built to want a clock** the ordering inverts"; "recency (a clock)" | R42 | **(i) POST-audit.** The doc was edited 09-11 without applying audit #2, and it never mentions the rewind. |
| 142 | "compositional motif transfer: **0.415 ± 0.096** vs 0.216" as a transfer headline | R38 | **(i) stale.** It is recipe-limited, and hierarchy-confounded (MapWM-Hier vs Plain-Flat). |
| 930 | "r=4 + weight sharing ... 0.986, 8/8 above 0.941 ... **the one super-additive pair** found here" | R41 | **(i) POST** (REFINE 08-31) |
| 38-41, 211-212, 447, 757-758 | rank "+0.085" / "at T=1024" with no extrapolation label | (b) | (ii). Qualified at 314-315 ("8x the training length"). |

Handled correctly: R21 (392-397), R22 (355-357), R23 (755-780), R27 (499-506), R28 (452-471), R29 (399-403, 864-869), R7 (84-86).

### mapformer_math.tex (working record, 09-07)

| line | phrase | R# | status |
|---|---|---|---|
| 2062 | "holding position far past the training length ... **or bound the accumulator explicitly, which is what a wrapped filter does**" | R9 | **(i) POST** (ACCUMULATOR 09-06). The review fixed the same table row (1068); this one was not fixed. |
| 2080-2084 | "they dominate the choice of positional encoding **by two orders of magnitude** (+0.461 ... against +0.003)" | R40, R40b, R19 | **(i).** This is a ratio claim, which R19 bans. |
| 1229-1232, 2426, 2446-2449 | +0.461 vs +0.003; "PoPE without path integration is ... a floor artifact" | R40, R40b | (i) |
| 910-925, 1253-1257, 2427 | "Why PoPE composes with an accumulated phase **and is inert with an index one**"; "PoPE alone is inert" | R40b | **(i) stale** |
| 48-53 | Abstract: "**geometry sets a hard threshold at r = D**" | R22 | (ii). Withdrawn in the body at 2359-2372, but still in the abstract. |
| 790-793 | +0.358 "no additive fallback ... outvoted" | R7, R43 | **(i) stale** |
| 1933, 2008-2010 | "mutually exclusive by construction"; the recency crossover framed as a clock task | R42 | (i) stale |
| 2065 | hierarchy "wins on compositional transfer" | R30 | (i) stale |

Handled correctly: R28 (1644ff), R22 body (2359-2388), R10/R11 (2490-2521), R20 (2477-2484),
R19 (1233-1237), R21 (589ff), R27 (1972-1995).

### report/report.tex and report_short.tex (quick grep pass only)

- **R40 (item a) is handled properly:** report.tex:586-592, 626 caption, 650-656, 769, 2127; report_short.tex:198-206.
- **R41 is handled:** report.tex:1682-1688; report_short:215, 483.
- **R23 is handled:** report.tex:1156, report_short:342.
- **R34 is handled:** report.tex:1998 ("solver mean").
- **R5/R6 is handled:** report_short:527-535 says the account is withdrawn.
- **Item (b), minor:** report_short.tex:290 and 336, and report.tex:998 and 1108, give "+0.085 at T=1024" with no "8x the training length" in the sentence. The caption at report.tex:1119 does say "trained at T=128".
- No live propagation of a listed item was found in this pass.

### .claude-memory/project_state.md (= memory/project_state.md, 09-23)

| line | phrase | R# | status |
|---|---|---|---|
| 15 | "The two positive results that survive are matched-length: **navigation +0.461 (T=128 -> 128)**" | R40 | **(i) POST.** PAPER2X2 is recorded in the same file at about line 162. |
| 216-218 | "**Index arms sit ON the measured 0.506 blank floor**; position +0.461, encoding +0.003" | R40, R40b | (i) |
| 134 | "at 200k PoPE 7/8 (**mean 0.965 vs paper 0.948**)" | R34 | (i) |
| 100, 144 (duplicated) | OOD levels "**sit AT a no-stack n-gram floor (0.884)**" | R37 | (i) |
| 137-140, 147-150 (four copies) | "MapPoPE-1L best arm, 0.927 ... **no interaction (+0.009, MDE 0.074)**" | R3 (F1 numbers) | (ii). There is an F1 warning at 80-81, but the Hewitt interaction is +0.065. |
| 219-221 | "Use r=4 ... +0.085 at T=1024" | (b) | (ii). The OOD audit is at 44-46. |
| 36-41 | "900-epoch pilot running" | none | Stale state: the 900-epoch batch landed 09-24, with r=4 8/8 solved, r=2 0/8, verdict unreadable. |

### Memory dir (quick pass)

| file:line | phrase | R# | status |
|---|---|---|---|
| project_robustness_vs_capability.md:10 | "navigation +0.461 (T=128 -> 128)" as a surviving matched-length positive | R40 | **(i) POST.** Line 13 also says "The **PoPE** paper's 'helps OOD' Dyck pattern"; it is the MapFormer paper's metric. |
| project_miniworld_flip_negative.md:26-28, 46 | Table headed "**converged effect**" includes "Torus +0.461 (index arms at the chance floor)" | R40 | (i). The row is not converged. |
| MEMORY.md:11 | "loop **raises the FLOOR**" | R41 | (i) |
| project_hierarchy_negative.md:151, 159-161 | "+0.315 super-additive"; "8/8 seeds >= 0.77 ... raises the floor" | R41 | (i) |
| reference_looped_transformer_lit.md:22-24 | super-additive +0.315; "benefit to the FLOOR" as the novelty claim | R41 | (i) |
| project_loop_and_correction.md:15-16 | "stabilisation and **token-type gating**, NOT inference" | R39 | (ii). Retracted at 48-54. |
| MEMORY.md:9; project_rank_and_selective_rope.md:16 | "matched-length test pending / not yet run" | none | Stale state: RANK_MATCHED has landed. |

Out of scope, but flagged by the brief: **CLAUDE.md**'s 09-23 top block carries the same "navigation +0.461 at T=128->128" framing.

## 3. Item (b): rank +0.085 presented without saying it is an extrapolation number

- **Unqualified:**
  - RESULTS_INDEX.md:147-148
  - positional_review.tex:1067, 1137-1138
  - mapformer_math.tex:48-53 (abstract), 2082, 2120 (column header "torus T=1024" only)
  - axes_measured.tex:38-41 (abstract), 211-212, 447, 757-758, 913-915
  - report_short.tex:290, 336; report.tex:998, 1108
- **Qualified:** axes_measured.tex:314-315; mapformer_math.tex:1498-1499, 1861; report.tex:1119 caption;
  memory reference_positional_landscape.md:39-40; project_rank_and_selective_rope.md:8-18; STORY.md:31.
- **Current evidence.** RANK_MATCHED e900: trained and tested at T=1024, r=4 solves 8/8 and r=2 0/8,
  +0.103 at T=1024, permutation p 0.0003. The registered verdict is UNREADABLE because 4 r=2 runs
  were still descending. This now gives partial matched-length support as a *learnability* result,
  not capacity: the rank-2 projection of r=4 reaches 0.9995. No document yet cites it.

## 4. Summary

Live propagations, counted as (i) rows. The (ii) count is in parentheses.

| document | (i) | (ii) |
|---|---|---|
| language_summary.html | 2 (+1 unverified superlative) | 1 |
| RESULTS_INDEX.md | 5 | 3 |
| BASELINE_TABLE.md | 2 | 2 (+1 scope note) |
| EM_WM_STATE.md | 0 | 4 |
| README.md | 2 | 2 (+1 minor) |
| report/STORY.md | 4 | 1 |
| positional_review.tex | 7 | 1 |
| axes_measured.tex | 6 | 1 |
| mapformer_math.tex | 7 | 1 |
| report.tex / report_short.tex | 0 (quick pass) | item (b) wording only |
| .claude-memory/project_state.md | 4 | 2 (+1 stale state) |
| memory dir, other files | 5 | 1 (+2 stale state) |

## The five most consequential

1. **language_summary.html:281, the live shared report.** It presents "+0.461 ... the index model
   sits on a 0.506 floor" as the matched-length navigation result, under the page's own
   "train at the length you test at" thesis. That number is the 16-epoch LinearLR recipe. Converged
   (PAPER2X2) it is **+0.243 with index RoPE at 0.805**. The project's own report.tex:626 says the
   index arms are undertrained under that recipe. The same framing is in project_state.md:15 and
   memory project_robustness_vs_capability.md:10.
2. **RESULTS_INDEX.md:22-38 (THE HEADLINE) and 154-156.** An n=3 table labelled "matched recipe"
   says "both index cells sit at the floor", "encoding ~0.003" and "PoPE is inert without path
   integration". Converged, the encoding effect is detectable at every length: -0.049, +0.114,
   +0.189. The same claims recur in BASELINE_TABLE.md:25/37, axes_measured.tex:46-47/139/418-423/448,
   and mapformer_math.tex:910-925/1253-1257/2080-2084. The last of these adds a
   "two orders of magnitude" ratio, the form R19 banned.
3. **positional_review.tex, the "presentable" review.** Four problems:
   - (a) It keeps the packing bound as the lower edge of the "useful range" (378-389) after DXR
     refuted it, and presents the critical-dimension account as "a mechanism for
     length-extrapolation failure" (1007-1018).
   - (b) It frames recency as "a task built to want a clock" on which "the ordering inverts"
     (148-150, 573-580, 1115-1119), and states the dichotomy as "mutually exclusive by construction".
     The 09-10 audit scoped both.
   - (c) It explains single-p0 > separate by "no additive fallback ... outvoted" (830-839), a
     withdrawn account.
   - (d) It says hierarchy "wins on compositional transfer" (1072) and recommends Flip-Flop as
     decisive (1193).
4. **axes_measured.tex, the results paper.**
   - Its abstract and transfer table present 16-epoch "index on the floor" numbers under a Setup
     paragraph that claims every run used 300 epochs with cosine decay at lr 1e-3.
   - It quotes "+0.461 (MDE 0.027)"; report.tex says no MDE was recorded.
   - It calls recency "a task built to want a clock", even though it was edited after the audit.
   - It sells "0.986 ... the one super-additive pair" (930) on paired statistics that REFINE showed
     unreliable.
5. **The wrap-bounds claim (R9), refuted 09-06, is live in two places.**
   - mapformer_math.tex:2062: "bound the accumulator explicitly, which is what a wrapped filter
     does". The review's copy of that same table row was fixed; this one was not.
   - README.md:279-280: "The wrap on innovation is what keeps θ̂ bounded at OOD sequence length —
     the key mechanism for length extrapolation".

Also notable:
- The html's "On text, code and music, plain PoPE still wins" (234) is contradicted by the page's
  own code row, by ABLATE (PoPE worse than RoPE on code) and by the repo's enwik8 numbers.
- The R41 loop-floor claim ("raises the floor", 8/8 >= 0.77) survives in MEMORY.md:11, in two
  memory notes, and in STORY.md:423.
