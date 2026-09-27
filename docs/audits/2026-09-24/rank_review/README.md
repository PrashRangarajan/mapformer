# Scripts and outputs behind `rank_review.md`

`../rank_review.md` (the 2026-09-23 end-to-end review of the rank matched-length line, behind
`RANK_MATCHED_PREREG.md` Amendment 2) names these files as "in this directory". They lived only in a
session scratchpad until 2026-09-27; copied here unchanged. Everything ran on CPU and touched
nothing in `runs/`. Paths inside the scripts are absolute to `/home/prashr/mapformer`.

| file | what it checks |
|---|---|
| `analyze_out.txt` | `python3 -m mapformer.analyze_rank_matched`, re-run |
| `check_strata.py` | `kinds()` against the env's own cells and revisit mask (400 trajectories) |
| `curves.py` | per-epoch losses, the 300-epoch T=1024 batch and the old T=128 batch |
| `stats.py` | t-based MDE, permutation / Wilcoxon / Fisher, ANCOVA, within-arm slopes |
| `strata_lossmatched.py` | per-stratum loss-matched contrasts and within-run composition |
| `project_r2.py`, `project_r2.txt` | existence check: a trained r=4 projected onto rank 2, loaded into an r=2 model, evaluated |
| `lagbins.py` | finer lag bins for stuck and trained seeds |
| `old_geometry.md` | geometry probe on the old T=128 checkpoints (written for the review, never a repo result) |
| `verify_proj/p.py`, `verify_proj/out.txt` | independent recheck of the rank-2 projection |

Not a results file: nothing here is citable on its own. The citable numbers are in
`RANK_MATCHED_RESULTS.md`, `RANK_PROJ_RESULTS.md` and, superseding both on the mechanism,
`RANK_SEP_RESULTS.md`.
