# The 2026-09-25 document correction pass, as scripts

How the five `.tex` documents were corrected on 2026-09-25 (commits `9915a50` axes_measured,
`360990d` positional_review, `d4c2f7e` mapformer_math, `bca4a47` report, `b4f762d` report_short).
These lived only in a session scratchpad until 2026-09-27.

`rep.py` is the tool and is the reusable part: `apply(path, pairs)` takes a list of
`(old, new)` string pairs and **exits with an error unless each `old` matches exactly once**. That
single property is what makes a large, citation-by-citation correction pass safe -- a silently
missed or doubly applied edit is the usual way these passes go wrong. Use it for the next pass.

| script | document |
|---|---|
| `ed_pr.py` | `positional_review.tex` |
| `ed_axes.py`, `ed_axes2.py` | `axes_measured.tex` |
| `ed_mm.py`, `ed_mm2.py` | `mapformer_math.tex` |
| `ed_rep.py` | `report/report.tex` |
| `ed_rs.py` | `report/report_short.tex` |

Usage as run: `python3 ed_pr.py <dir containing rep.py>` from the repo root, then rebuild the PDF.

The edits themselves are historical -- they are already in the files and in git. Note that this
pass predates `DYCK_MDEPTH_RESULTS.md` and `RANK_SEP_RESULTS.md`, so several of the replacements it
installed (the "not separated" rank caveat above all) are themselves now stale. The list of what
still needs fixing is in `docs/WHERE_THINGS_STAND.md`, under "Known stale".
