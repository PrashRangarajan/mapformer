import sys; sys.path.insert(0, sys.argv[1]); from rep import apply
P = [
("""# Results index (regenerated 2026-09-24; rank, documents and statistics updated 2026-09-25)""",
"""# Results index (regenerated 2026-09-24; rank, documents and statistics updated 2026-09-25; rank separation, Dyck matched depth, loop-rank and catalogue updated 2026-09-27)"""),
("""(record), `report/report.pdf` and `report/report_short.pdf`; all brought into line with this index on
2026-09-25.""",
"""(record), `report/report.pdf` and `report/report_short.pdf`; all brought into line with this index on
2026-09-27 (rank separation, Dyck matched depth, torus loop-rank). `report/language_summary.html` (the
shared report) was not part of that pass."""),
# rank row
("""| Shared r=4 finds the torus solution where r=2 does not (trained and tested at T=1024, 900 ep, matched initialisation; why -- per-head rank, cross-head sharing or W_out scale -- is unseparated) | SOLVED within 900 ep: our shared r=2 **0/8**, a per-head r=2 **2/8**, shared r=4 **8/8** (Fisher p 0.0002 / 0.007); accuracy 0.894 / 0.885 / 0.998. A rank-2 projection of each solved r=4 scores 0.9955 frozen and is held under training (7/8 vs the r=4 control's 7/8): a SEARCH deficit, in the paper's own design | 8 | constant floor 0.506 (wrap-only 0.507) | SOLID, budget-scoped (within 900 epochs); r=4's smaller `W_out` init (bound 0.5 vs 0.707) is unseparated; one task, one width; MapWM family only | `RANK_MI_RESULTS.md`, `RANK_PROJ_RESULTS.md`, `RANK_MATCHED_RESULTS.md` |""",
"""| It is the PER-HEAD rank of the content-to-angle map that decides whether training finds the torus solution (trained and tested at T=1024, 900 ep, every arm built from our r=2's initial weights) | SOLVED within 900 ep, five arms: per-head rank 2 -- our shared r=2 **0/8** (0.894), per-head r=2 **2/8** (0.885), block-diagonal r=4 **2/8** (0.948); per-head rank 4 -- shared r=4 **8/8** (0.998), per-head r=4 **8/8** (0.999). Per-head rank FIRES (D - C_bd, Fisher and perm p 0.0070, Holm 0.028); sharing (D - C, Fisher 1.00, perm 0.59) and W_out per-entry scale (C_bd - B, Fisher 1.00, perm 0.24) UNMEASURED. A rank-2 projection of each solved r=4 scores 0.9955 frozen and is held under training (7/8 vs the r=4 control's 7/8): a SEARCH deficit | 8 | constant floor 0.506 (wrap-only 0.507) | SOLID, budget-scoped (within 900 epochs); r=4's smaller `W_out` init (bound 0.5 vs 0.707) was tested (C_bd - B) and is UNMEASURED, not shown to matter; low initial angle scale is not necessary for failure (A 0.335 / B 0.328 fail like C_bd 0.208); n_heads=2, one task, one length, one recipe; MapWM family only | `RANK_SEP_RESULTS.md`, `RANK_SEP_PREREG.md`, `RANK_MI_RESULTS.md`, `RANK_PROJ_RESULTS.md`, `RANK_MATCHED_RESULTS.md` |
| Search aids partly recover rank 2 but do not reach rank 4 (torus, T=1024, 900 ep) | r=2 + loop x4 (bit-identical params and init to r=2) 2/8 solved, 0.973 (perm p 0.0034 vs r=2); r=2 at 4 real layers 5/8, 0.990 (Fisher 0.0256, perm 0.0012); plain r=2 0/8, 0.894; r=4 8/8, 0.998. 4 real layers beat the loop +0.017 (perm p 0.0009). Final losses form three regimes; no rank-2 run enters r=4's | 8 | constant floor 0.506 | registered H1 verdict UNMEASURED; the 0.05 solved cutoff falls inside the aids' spread (at 0.08 H1's condition would have been met), so uncertain rather than negative; depth (4x params) beats the matched-parameter loop, so NOT search at constant capacity | `LOOP_RANK_RESULTS.md`, `LOOP_RANK_PREREG.md` |"""),
# Dyck: add matched-depth row after the D4 ladder row
("""| Dyck-2 depth ladder at the training cell (L32 D4), width fixed | position +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8); index RoPE 0.979 at 4L | 8 | chance 0.5 | SOLID (matched length and depth) | `DYCK_LADDER_RESULTS.md` |""",
"""| Dyck-2 depth ladder at the training cell (L32 D4), width fixed | position +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8); index RoPE 0.979 at 4L | 8 | chance 0.5 | SOLID (matched length and depth) | `DYCK_LADDER_RESULTS.md` |
| Dyck-2 at matched depth: path integration is worth ~3 layers of attention (trained AND tested at L32 D12, A2f) | position main +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers (8/8 each, 1x budget). Mixture training (D in 4..12, 4L) keeps +0.110 at L32 D12 (8/8) and +0.043 at D4; L128 D12 unmeasured | 8 | chance 0.500; best floor 0.594 | SOLID as depth-substitution (parameter efficiency), not as something depth cannot buy: at 4L and 3x budget every arm is at ceiling (index 0.997-0.998, path 1.000), effect +0.002 | `DYCK_MDEPTH_RESULTS.md`, `DYCK_MDEPTH_PREREG.md` |"""),
# paper replications row: Hewitt cell
("""(8/8, F1; on Hewitt closing accuracy +0.064, 0.638 vs 0.574 at L128 D12, same direction); levels do not replicate""",
"""(8/8, F1; on Hewitt closing accuracy +0.064, 0.638 vs 0.574 at L128 D12, same direction -- that cell is 3x the training depth and 4x its length, i.e. depth-extrapolation, and the 4L depth effect closes at matched depth, `DYCK_MDEPTH_RESULTS.md`); levels do not replicate"""),
("""| 8 / 8 / 5 | F1 no-stack 0.884 | SOLID | `DYCK_RESULTS_bs128.md`,""",
"""| 8 / 8 / 5 | F1 no-stack 0.884 | SOLID as replications; the Dyck ordering is read OOD in depth (robustness) | `DYCK_RESULTS_bs128.md`,"""),
# remove NEEDS CONTROL Dyck row
("""| Dyck ladder at L32 D12 | +0.290 / +0.209 / +0.159 / +0.168 at 1-4L (8/8); index plateaus 0.76-0.78 | matched LENGTH but **3x the training depth**: a matched-depth control, and an index-arm budget extension ("stop climbing" rests on a slope rule read after LR decay). The results file now carries a correction banner (2026-09-24) | `DYCK_LADDER_RESULTS.md` |
""",
""""""),
# Open: rank old numbers
("""  restart) are UNREADABLE (four r=2 runs still descending); `RANK_MI_RESULTS.md` is the readable test.""",
"""  restart) are UNREADABLE (four r=2 runs still descending); `RANK_SEP_RESULTS.md` (with `RANK_MI_RESULTS.md`)
  is the readable test, and it separates the cause: per-head rank."""),
]
apply("RESULTS_INDEX.md", P)
Q = [
("""**Live negatives** (do not re-run): CLAUDE.md, "Live negatives". **Withdrawn** (do not cite):
CLAUDE.md, "Withdrawn -- do not cite", plus `archive/void/` (54 files, each bannered) and
`archive_stale/` (35 files).
""",
"""**Live negatives** (do not re-run): CLAUDE.md, "Live negatives". **Withdrawn** (do not cite):
CLAUDE.md, "Withdrawn -- do not cite", plus `archive/void/` (54 files, each bannered) and
`archive_stale/` (35 files).

Closed 2026-09-26 (formerly NEEDS CONTROL here; on CLAUDE.md's withdrawn list): the Dyck ladder's
position effect at L32 D12 (+0.168 at 4 layers, trained at D4) as a capability result, and "the index
arms plateau / depth closes 40% of the gap then stops". Trained at D12, every 4-layer arm reaches ceiling
at 3x budget (+0.002); the 1x budget limits the index arms (3x - 1x = +0.021, 8/8). What survives is the
matched-depth row above (`DYCK_MDEPTH_RESULTS.md`).
"""),
]
apply("RESULTS_INDEX.md", Q)
