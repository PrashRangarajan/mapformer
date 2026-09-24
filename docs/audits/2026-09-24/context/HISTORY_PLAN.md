# Where the chronological log goes, and what would be lost

## Plan (nothing is deleted)

1. `mkdir -p docs && git mv CLAUDE.md docs/LOG.md`, then prepend one header line:
   `# MapFormer project log, 2026-04..2026-09-24 (was CLAUDE.md; not auto-loaded; newest first at the top, oldest after "START HERE")`.
   Do NOT name it `docs/CLAUDE.md` (subdirectory CLAUDE.md files are loaded when files there are read).
   Keeping the file verbatim means every fact dropped from the auto-loaded file is preserved
   with its git history (`git log --follow docs/LOG.md`).
2. Write the new `CLAUDE.md` from `CLAUDE_proposed.md`.
3. Future sessions: narratives go to `project_state.md` (live) and the results files; a
   session that wants a diary appends to `docs/LOG.md`, never to CLAUDE.md.
4. Move the three dated blocks that `project_state.md` duplicates (2026-09-12..15,
   2026-09-15..20, 2026-09-21) out of `project_state.md`; they are already in `docs/LOG.md`.
5. Regenerate `RESULTS_INDEX.md` (last 2026-09-11) so it covers Dyck, Bach, Indirect, code,
   PoPE ablation and rank-matched, drop its separate rule list (it duplicates and mis-numbers
   CLAUDE.md's), and point to CLAUDE.md for rules.

## Facts that exist ONLY in CLAUDE.md (VERIFIED by grep over every top-level *.md,
## .claude-memory/*.md, archive/void/*.md and archive_stale/*.md; code checked where noted)

Kept in `CLAUDE_proposed.md` (so they survive in the auto-loaded file too):
- setsid for anything > ~2 min; foreground tool calls SIGTERM'd at 2 min with their children; `run_in_background` jobs die with the session (0 other carriers).
- `wait` returns regardless of child success; judge by the artifact (0).
- Atomic replacement of a running driver via `mv` (bash keeps the old inode through fd 255) (0).
- `local a=$1 b=$2 c="...$b"` under `set -u` expands before assigning (0).
- `--fast-attn`: TF32 breaks equivalence on the compositional trainer (logit diff 6.0e-01, grad cos 0.99995) and SDPA is invalid for MapEM's Hadamard product (the 6.0e-01 figure has 0 carriers; SDPA/MapEM is mentioned only in passing in 4 preregs).
- MapFormer v4 Table 2 targets (MapWM-r2 1.00/1.00/0.99, MapEM-os/s 1.00/1.00/1.00) and "ours matched v1, marginally below v4" (0 carriers; `README.md` and the OOD files never mention v4).
- The resolved Dyck `base=10000` confound pointer (numbers are in `DYCK_DEPTH_RESULTS.md`, but only CLAUDE.md connects them to the "UNCONTROLLED CONFOUND" claim it made on 2026-09-21).

Preserved only by `docs/LOG.md` under the plan -- decide whether each needs a better home:
- **Run-directory map for the Dyck / Indirect / Bach / torus / recency line** (`runs/dyck_bs128`, `runs/dyck_t2`, `runs/dyck_cross`, `runs/dyck_decay`, `runs/dyck_lam`, `runs/dyck_t3`, `runs/indirect*`, `runs/jsb*`, `runs/jsb_base{512,8192,32768}`, `runs/torus_t3`, `runs/recency_t3`). CLAUDE.md itself says "none are named in the results files for the Bach line"; only `T1_PREREG.md` names `runs/jsb_len512`. **Recommend: copy into `project_state.md` or a `RUNS_MAP.md`.**
- **"`DYCK_T3_RESULTS.md` is an EMPTY artefact of a mis-set analysis script -- ignore it"** (0 carriers). Recommend a one-line banner inside that file.
- The rest of the v4 diff: Mamba 0.42/0.77/0.40 -> 0.38/0.66/0.30, MAmPa 0.74/0.93/0.60 -> 0.84/0.96/0.71, RoPE(4L) 2D IID 0.33 -> 0.82, TAPE and PathAtt added as baselines, MapWM split r1/r2 (0 carriers for the v4 Mamba/MAmPa numbers). Recommend a short `PAPER_V4_DIFF.md`.
- `--fast-attn` is 2.56x faster at 37% memory on the torus (0 carriers for "2.56x"; `FORGET_CONTROL.md` mentions `USE_SDPA`).
- The 2026-09-06 review history: two referees returned MAJOR REVISION; an adversarial audit found 13 errors in the theory note (0 carriers). Genuinely historical.
- The April-May session narratives (Level 1/1.5/2, PC v1-v4, DoG, MiniGrid wrapper design, orchestrator scripts, "Filesystem map" of `figures_*` dirs, "Quick reproducibility commands" for the InEKF-era scripts). Their results live in `RESULTS_PAPER.md`, `DETAILED_RESULTS.md`, `README.md`, `archive_stale/`, `archive/void/` and the code (e.g. `log_R_init_bias` is in `README.md`, `DETAILED_RESULTS.md`, `MINIGRID_EM.md`); the narrative itself is obsolete.
- `RESULTS_LEVEL2.md`, `RESULTS_LEVEL15.md`, `RESULTS_LEVEL15_CLEAN.md` and `STOCHASTIC_TRANSITION_RESULTS.md` are cited by CLAUDE.md but do not exist anywhere (CLAUDE.md already says the first three were superseded by `RESULTS_PAPER.md`; the fourth was never produced). Obsolete.

Everything else sampled (65 specific numbers and rules across all eras) has at least one other
carrier: a results file, a memory file, `RESULTS_INDEX.md`, `README.md` or the code. CLAUDE.md
names 142 distinct `*.md` files; all exist at top level except 11 in `archive/void/`, 1 in
`archive_stale/` (`DOG_RESULTS_FIXED.md`), the 4 that exist nowhere (above), and the 3 April
`paper/0x_*.md` drafts (`paper/` now holds `main.tex`).
