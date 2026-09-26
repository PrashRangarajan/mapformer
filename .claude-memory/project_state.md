---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md.
metadata:
  type: project
---

Written 2026-09-24 ~23:00. Goes stale fast: check `git log`, `.done` markers and results files first.
Not pushed (`main` well ahead of `origin/main`).

## Running
Nothing.

## Last results (all committed)
- **Dyck CLOSES at matched depth** (`DYCK_MDEPTH_RESULTS.md`, 210 runs, 2026-09-25): trained at L32 D12,
  all 4-layer arms at ceiling (index 0.997-0.998, path 1.000, floor 0.594), effect +0.002. The ladder's
  +0.168 was depth extrapolation from D4. Survives: depth-substitution (+0.353 1L -> +0.024 4L) and
  mixture training over D 4..12 (+0.110). Closed in the same batch: the RoPE base confound at 4L; the
  1x ladder budget limits index arms (+0.021).
- **Rank resolves to PER-HEAD rank** (`RANK_SEP_RESULTS.md`): rank 2 per head 0/8, 2/8, 2/8; rank 4 per
  head 8/8, 8/8. Sharing (D-C) and W_out scale (C_bd-B) unmeasured; initial angle scale excluded.
  Supersedes the "unseparated" caveat in `RANK_MI_RESULTS.md`.
- Shared report at v8 (Dyck headline replaced by depth-substitution; rank section updated).
- Earlier: the 2026-09-24 audits and code fixes (`docs/audits/2026-09-24/`), CLAUDE.md 183 KB -> 23 KB.

## Waiting on the user
1. The .tex documents and RESULTS_INDEX still carry the OLD Dyck framing (+0.168 at 3x depth as the
   language line's positive result) and the "unseparated" rank caveat. Both are now superseded -- a
   correction pass like 2026-09-25's is needed.
2. Experiments, cheapest first: code checkpoints rescored on the full val file (eval-only, 1-2 GPU-h; settles
   the code "reversal", p ~0.1); Dyck control at matched nesting depth (~3 GPU-h); sign at matched length
   (10-20 GPU-h); rank-2 at r=4's W_out init scale (~2 h).

## Known loose ends
- md5 guards abort resuming any pre-2026-09-24 series (training files changed bit-identically); regenerate
  `code_md5.txt` deliberately to resume.
- `analyze_dyck_depth.py` pairs arms by list position (same bug fixed in the ladder); committed numbers fine.
- Eight results files sit in /home/prashr, outside the repo (`MINIWORLD_GATES.md` exists only there).
- Jobs per GPU not re-measured with `--fast-attn`.
