# The 2026-09-27 document correction pass, as scripts

Brings the five `.tex` documents and `RESULTS_INDEX.md` into line with `RANK_SEP_RESULTS.md`,
`DYCK_MDEPTH_RESULTS.md` and `LOOP_RANK_RESULTS.md`. `rep.py` is copied from the 2026-09-25 pass.
Usage as run: `python3 ed_X.py docs/prepared/tex_edits_2026-09-27` from the repo root. The edits are
already applied; a few small follow-ups after the end-to-end read are noted as comments at the foot of
`ed_rep.py` and `ed_pr.py`.

| script | file |
|---|---|
| `ed_pr.py` | `positional_review.tex` |
| `ed_axes.py` | `axes_measured.tex` |
| `ed_mm.py` | `mapformer_math.tex` |
| `ed_rep.py` | `report/report.tex` |
| `ed_rs.py` | `report/report_short.tex` |
| `ed_index.py` | `RESULTS_INDEX.md` (hand-written rows; the catalogue was regenerated with `docs/tools/catalog_results_index.py`) |
