---
name: reference-review-documents
description: The documents (review, results paper, record, reports), the live shared report link and its republish rule, and the 40-paper corpus in papers/.
metadata:
  type: reference
---

| file | what it is |
|---|---|
| `positional_review.pdf` | **the presentable one, 21pp.** Arena review: classification, log-polar unification, clock/map as a conceptual consequence, **the relational map** (what the user most wants to show), MapEM/TEM placements, the memory reading. No measurements. |
| `axes_measured.pdf` | **the results paper, 17pp.** Factorisation-and-transfer (the thesis), sign / rank / alpha / the crossover, Flip-Flop, the gated null, where MapPoPE lands. |
| `mapformer_math.pdf` | the working record, 38pp: everything plus every correction and retraction. |
| `report/report.pdf`, `report/report_short.pdf` | 43pp / 10pp paper-style reports of surviving results. |
| `report/language_summary.html` | source of the live shared report (below). |
| `papers/` | **40 sources, all read first-hand.** `txt/<key>.txt` tracked and greppable; `pdf/` gitignored (69 MB), restored by `papers/fetch.sh`; `papers/INDEX.md` records the claim each reading checks. |

The review/record split was made 2026-09-08 because the review had grown to 81% of the record. All
five were brought into line 2026-09-27 (rank separation, Dyck matched depth, loop-rank) and corrected
2026-09-30 where later results contradicted them (sign at matched length, code full-val). They are
incomplete, not wrong: none carries rank 3, H1 part 1, H3, the text world, the context step, 3D rank /
wrap, what/where or the leak (`docs/WHERE_THINGS_STAND.md`, "Known stale"). Build from source.

## The shared report

**Live link:** https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc ("Position Codes on Language", v10 on
2026-09-30, anyone with the link; viewers see updates immediately; lacks 3D rank / wrap, what/where, leak). Source of truth
`report/language_summary.html`, byte-identical to what was published.
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
