---
name: reference-review-documents
description: The documents (current drafts 2 and 6 and their PDFs; the stale September review, results paper, record, reports), the live shared report link (v22) and its republish rule, and the paper corpus in papers/ (44 + 28).
metadata:
  type: reference
---

**Updated 2026-10-29 (cleanup C6).** The current write-up is **main paper draft 2** (`docs/paper/MAIN_DRAFT2.md` +
`MAIN_APPENDIX2.md`; PDFs `docs/paper/pdf/main_paper_draft2{,_appendix,_with_appendix}.pdf`) and the **sign companion,
draft 6** (`docs/paper/DRAFT6_workshop.md` + `APPENDIX_draft6.md`; `4_sign_workshop_paper.pdf`, `5_sign_workshop_appendix.pdf`;
theory appendix `APPENDIX_theory_v3.md`), with the claim map `docs/paper/CLAIM_MAP_clock_map.md` and the review
`docs/audits/2026-10-28/FULL_REVIEW.md`. Rebuild: `python3 docs/paper/tools/build_pdfs.py docs/paper/pdf`. Superseded drafts:
`docs/paper/old/`. The table below is the September set: STALE (stops at 2026-09-30), kept for the record.

| file | what it is |
|---|---|
| `positional_review.pdf` | **the presentable one, 21pp.** Arena review: classification, log-polar unification, clock/map as a conceptual consequence, **the relational map** (what the user most wants to show), MapEM/TEM placements, the memory reading. No measurements. |
| `axes_measured.pdf` | **the results paper, 17pp.** Factorisation-and-transfer (the thesis), sign / rank / alpha / the crossover, Flip-Flop, the gated null, where MapPoPE lands. |
| `mapformer_math.pdf` | the working record, 38pp: everything plus every correction and retraction. |
| `report/report.pdf`, `report/report_short.pdf` | 43pp / 10pp paper-style reports of surviving results. |
| `report/language_summary.html` | source of the live shared report (below). |
| `papers/` | **44 sources in the main corpus, all read first-hand, plus 28 from the 2026-10-14..16 checks in `papers/new_*/`.** `txt/<key>.txt` tracked and greppable; `pdf/` and `new_*/**/*.pdf` gitignored, restored by `papers/fetch.sh` / each folder's `SOURCES.txt`; `papers/INDEX.md` records the claim each reading checks. |

The review/record split was made 2026-09-08 because the review had grown to 81% of the record. All
five were brought into line 2026-09-27 (rank separation, Dyck matched depth, loop-rank) and corrected
2026-09-30 where later results contradicted them (sign at matched length, code full-val). They are
incomplete, not wrong: none carries rank 3, H1 part 1, H3, the text world, the context step, 3D rank /
wrap, what/where or the leak (`docs/WHERE_THINGS_STAND.md`, "Known stale"). Build from source.

## The shared report

**Live link:** https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc ("Position Codes on Language", **v22 on
2026-10-11** per CLAUDE.md -- rank tests added; anyone with the link; viewers see updates immediately; lacks everything after
2026-10-11). Source of truth `report/language_summary.html`. A second artifact, JmzdmkFup4V3kV3C5YndPm (navigation told in
words), has source `report/navigation_in_words.html` (last edited 2026-10-05).
- Edit the HTML, then publish with the Artifact tool passing `url=` that link. A publish without
  `url` creates a NEW link and the one the user shared goes stale. In a new session `action: "read"`
  the URL first (the tool refuses a publish to an artifact the session has not read).
- Commit the HTML after every republish so the repo never lags the live page.
- Structure (user's preference): short version first, positive results first, failures briefly at the
  end, every term defined. Content follows CLAUDE.md's citable / withdrawn tables.

## The corpus

Grep it instead of re-searching the web: `grep -n -i "data-dependent" papers/txt/hgrn.txt`. Keys
include mapformer srope pope grape mamba3 fox rope alibi xpos nope carope jordan_rope liere alg_pe
puranik_janestreet pj_rope mamba mamba2 gla path deltanet rwkv7 cope tape mesanet titans hgrn hgrn2
sarrof grazzi dape and five surveys (`ls papers/txt`: 44 files, including `tale_two_algorithms`,
Whittington et al., Neuron 2025, added by hand).

## Rules the documents are held to

- **No meta-narrative in the presentable one** ("an earlier version proposed", "corrected
  2026-09-..." belong in the record).
- **The relational map's bold means one thing**: a value held by at most 3 of the 18 rows, declared in
  the caption. No row name is emphasised (bolding MapFormer's row read as advocacy).
- **MapPoPE is marked as this work's construction**; every other row is published.
- The relational map is **Table 4** in the review.

## Positioning, measured not asserted

Five surveys checked; four have no content-dependent phase mechanism. The fifth (Zhang et al.
2503.17407) HAS the category ("content-aware position embedding": CoPE, DAPE) and predates most of
it. **Do not write "no survey covers this."** The cell was opened on the ADDITIVE side. 2511.08243 is
WITHDRAWN; do not chase it. See [[reference-positional-landscape]].
