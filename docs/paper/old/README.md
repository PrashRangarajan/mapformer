# Superseded paper documents

Moved here on 2026-10-29 (cleanup step C4) with `git mv`; history follows (`git log --follow docs/paper/old/<file>`).
The current documents stay in `docs/paper/`: draft 2 (`MAIN_DRAFT2.md`, `MAIN_APPENDIX2.md`, `MAIN_DRAFT2_notes.md`),
draft 6 (`DRAFT6_workshop.md`, `APPENDIX_draft6.md`, `DRAFT6_changes.md`), `APPENDIX_theory_v3.md`, the claim map
(`CLAIM_MAP_clock_map.md`) and `BIG_PICTURE_2026-10-24.md`. Their PDFs are in `docs/paper/pdf/` (rebuild:
`python3 docs/paper/tools/build_pdfs.py docs/paper/pdf`).

| here | what | replaced by |
|---|---|---|
| `MAIN_DRAFT1.md`, `MAIN_APPENDIX1.md`, `MAIN_DRAFT1_notes.md` | main paper, draft 1 | draft 2 |
| `OUTLINE_main_v2.md`, `OUTLINE_overall.md`, `OUTLINE_workshop.md` | outlines of drafts 1-5 | draft 2, draft 6, `BIG_PICTURE_2026-10-24.md` |
| `DRAFT_workshop.md`, `DRAFT3_*`, `DRAFT4_*`, `DRAFT5_*`, `APPENDIX_draft5.md` | sign workshop paper, drafts 2-5 | draft 6 |
| `APPENDIX_theory.md`, `APPENDIX_theory_v2.md` | theory appendix v1, v2 | `APPENDIX_theory_v3.md` |
| `READ_draft4.md`, `MOCK_REVIEW_draft3.md` | end-to-end read and mock review of drafts 3-4 | `docs/audits/2026-10-26/CRITICAL_REVIEW.md`, `docs/audits/2026-10-28/FULL_REVIEW.md` |
| `VENUES_2026-10-15.md` | venue plan of 2026-10-15 | `.claude-memory/project_state.md` |
| `pdf/` | 14 PDFs of draft 1, its outline, expansion notes and two results notes, and a 2026-10-19 sources zip (cleanup C3) | `docs/paper/pdf/` |

Redirect stubs. Five pre-registrations cite four of these files at their old paths, and pre-registrations are never
edited, so a short stub stays at each old path and points here:

| stub at | cited by |
|---|---|
| `docs/paper/MAIN_DRAFT1.md` | `SROPE_TORUS_PREREG.md`, `TORUS_LRSWEEP_PREREG.md` |
| `docs/paper/DRAFT5_workshop.md` | `LRSWEEP_PREREG.md` |
| `docs/paper/READ_draft4.md` | `ROBUST_PREREG.md` |
| `docs/paper/MOCK_REVIEW_draft3.md` | `GAUGE_PREREG.md` |

`docs/plans/2026-10-11_next_gpu.md` and `docs/plans/2026-10-14_next_steps.md` (cited by RANK_TESTS, ROBUST and
TORUS_START pre-registrations) were left in place.
