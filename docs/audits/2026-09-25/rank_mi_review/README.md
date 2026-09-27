# Review of the matched-initialisation rank batch (`RANK_MI_RESULTS.md`), 2026-09-25 00:43-00:51

Read-only review of `runs/rank_mi` (arms A `Vanilla` shared r=2, B `Vanilla_r2ph` per-head r=2,
C `Vanilla_r4mi` shared r=4). It lived only in a session scratchpad until 2026-09-27; copied here
unchanged, minus the 20 MB of replay checkpoints. `.log` files are renamed `.txt` because `*.log`
is gitignored.

| file | what it checks |
|---|---|
| `replay_epoch1.py` | replays EPOCH 1 of every `rank_mi` run with the current code and compares the first-epoch loss with the stored checkpoint's `losses[0]`, bitwise; plus construction-level matched-init checks and the initial Delta scale per arm |
| `replay.txt`, `replay2.txt`, `replay3.txt`, `replay_epoch1.json` | its output. All 24 runs `equal: True`. Matched init confirmed at construction: `Vanilla_r2ph vs Vanilla` and `Vanilla_r4mi vs Vanilla` max weight diff 0.0 on every seed, `keys equal` true, `r2ph.w_in == r4mi.w_in` |
| `analysis_rerun.txt` | `analyze_rank_mi` re-run: SOLVED 0/8, 2/8, 8/8; T=1024 accuracy 0.894 / 0.885 / 0.998; C-A Fisher 0.0002, C-B 0.0070; reproduction of A seed 3 bit-exact over 900 epochs |

The initial angle scales recorded here (`delta_std` 0.28-0.41 across all three arms, same range per
seed) are what later let `RANK_SEP_RESULTS.md` rule out initial angle scale as the explanation.

Not a results file. The citable numbers are in `RANK_MI_RESULTS.md`; `RANK_SEP_RESULTS.md`
supersedes its "unseparated" caveat.
