# Second rank review, 2026-09-24 morning: does the rank-2 solution HOLD?

Follow-up to `../rank_review.md`, run while the matched-initialisation batch trained. It is the
evidence behind "a rank-2 solution EXISTS (0.9955) and is HELD under training", which CLAUDE.md and
`RANK_PROJ_RESULTS.md` both cite. It lived only in a session scratchpad until 2026-09-27; copied
here unchanged, minus the two snapshot checkpoint directories (30 MB of `.pt`). `.log` files are
renamed `.txt` because `*.log` is gitignored.

| file | what it checks |
|---|---|
| `proj_rerun.txt` | frozen projected r=2 vs its source r=4 at T=1024, per seed: mean 0.9966 -> 0.9955, >= 0.95 on 8/8; then trainable vs control vs the r=2 continuation |
| `check_init.py` | the projected, continued and control arms really start from the weights they claim to |
| `curves.py`, `curves.json` | per-epoch losses for `rank_matched_e900` and the `e900c` warm restart, both arms, 8 seeds |
| `e900c_rerun.txt`, `e900c_status.txt` | the committed `e900c` analysis regenerated; `IDENT` = byte-identical |
| `snap_train.py` | re-runs the continuation recipe saving state every N epochs (review probe, not a repo script) |
| `probe_snaps.py`, `probe_ref.txt`, `probe_s0.txt`, `probe_s6.txt`, `snap_proj_s0.txt`, `snap_proj_s6.txt` | action-code geometry and T=1024 strata accuracy along the snapshot trajectory of two seeds |
| `reeval.py` | re-scores the three committed rank JSONs from their checkpoints |

Not a results file. `RANK_SEP_RESULTS.md` (2026-09-25) supersedes this line's interpretation: the
factor is per-head rank.
