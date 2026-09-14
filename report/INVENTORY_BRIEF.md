# Inventory brief (shared by all inventory agents)

Goal: a paper-style LaTeX results report of every SURVIVING experiment in this repo
(/home/prashr/mapformer; MapFormer, Rambaud et al. 2025, arXiv:2511.19279, plus ~5 months of
follow-ups). Retracted, voided and withdrawn results are OMITTED from the report, but you must
know them in order to exclude them. Your job is one experiment line: read its files first-hand
and write a structured inventory that the report writer can use without re-reading sources.

Research goal that frames the report: positional encoding as the mechanism by which a model
learns a relational "where" kept separate from the "what" (TEM's factorisation-buys-transfer
claim). Every evaluation redraws the observation map, so evaluations are transfer measurements.

## Rules
- Read-only on the repo EXCEPT your one output file. No training, no GPU, no git.
- Before writing, read for exclusions: `RESULTS_INDEX.md` (whole), `N3_AUDIT.md`, `KNOWN_BUGS.md`,
  `archive/void/README.md`, and grep `CLAUDE.md` for CORRECTED / RETRACTED / WITHDRAWN / VOID
  blocks touching your files. A CORRECTED or AUDIT block at the top of a results file supersedes
  its body. Anything in `archive/void/` is out. lm200 rows from April are out; clean/noise rows of
  the same era are valid unless marked otherwise.
- Copy numbers EXACTLY from the source; name the source file for every number. Never compute a new
  number and present it as a result. If two files disagree, the later/CORRECTED one wins; record
  the disagreement.
- Statistics language: MDE = 2.8*sd/sqrt(n). Below MDE = "unmeasured", never "null". A same-seed
  rerun is determinism, not replication. n<=3 results are at most EXPLORATORY unless a later
  file re-ran them at higher n.
- Mechanism claims: keep "associated with" vs "established by intervention" distinct; keep
  "mechanism unidentified" where the files say so.

## Output format: report/inventory/<LINE>.md

Start with a 5-10 line overview of the line: the question it asked and what survived.

Then one block per experiment (merge files that are one experiment, e.g. PREREG + RESULTS + GATES):

```
### <ID> <short title>
- Dates: <from file / git log -1 --format=%ad --date=short -- <file>>
- Question:
- Task / environment: <incl. chance or measured floor, train length, eval lengths, held-out map?>
- Arms: <names as in code, params if stated>
- Seeds / batch: <n per arm; one batch or not; recipe (epochs, lr, schedule)>
- Validity gates: <shortcut n-gram gates, context destruction, convergence / rule 9 r(loss,acc), repro controls>
- Result: <compact table or bullets; delta, sd or MDE, seeds+, verdict, copied exactly>
- Status: CITABLE (detectable, adequately powered) | DIRECTIONAL (right sign, under MDE) |
  POWERED NEGATIVE (MDE smaller than the effect being dismissed) | EXPLORATORY | PRIOR-ART REPLICATION
- Pre-registered? yes (file) / no; if yes, which predictions held or failed
- Caveats the report must carry:
- Sources: <files>
- Bears on: <which claim about the where/what, rank, sign, EM vs WM, correction, hierarchy, loops, environment...>
```

End with three sections:
1. `## Excluded` — table of files/claims in your line that are retracted/void/withdrawn, one-line
   reason and the file that killed each (for the writer's record; not for the report).
2. `## Cross-line dependencies` — results in your line that other lines' claims rely on.
3. `## Files read` — every file you opened; and `## Files in scope not covered` if any.

Aim for completeness over prose. Typical length 2,000-6,000 words.
