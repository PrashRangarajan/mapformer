---
name: feedback-git-add-silent-failure
description: A failed `git add` stages NOTHING; with stderr suppressed it is invisible, and a leading space in `git status --short` means unstaged.
metadata:
  type: feedback
---

`git add -A papers INDEX.md .gitignore mapformer_math.tex ...` failed because
`INDEX.md` was at `papers/INDEX.md`, not the repo root. **A failed `git add`
stages nothing at all** — not even the pathspecs that did match. With `2>/dev/null`
on the command the `fatal:` line was invisible, and the follow-up `git add papers`
picked up only the corpus. The commit shipped 28 papers without the 26-page
document they exist to support, and the PDF in git stayed at the pre-session build
for two more commits.

**Why the check didn't catch it:** in `git status --short`, column 1 is the INDEX
and column 2 the WORKING TREE. `M ` (M then space) is staged; ` M` (space then M)
is modified-but-unstaged. I printed the list, saw `M` on every line I cared about,
and read them as staged.

**Why:** this is the same failure family as two entries already in CLAUDE.md — an
unconditional `echo pushed` masking a failed commit, and `grep -c` returning 0
exiting nonzero and breaking an `&&` chain. Suppressing stderr on a command whose
failure mode is silence, then trusting a summary instead of the artifact.

**How to apply:**
- Never `2>/dev/null` a `git add`, and never batch pathspecs you have not verified
  exist. `git add -A <dir>` and explicit file lists are fine; a typo'd path kills
  the whole call.
- Verify the ARTIFACT after committing, not the absence of a crash:
  `git show HEAD:<file> | ...` and `git status --short --untracked-files=no`
  (empty output = clean). Same rule as [[feedback-verify-before-relaying]] and the
  `wait`-returns-regardless lesson.
- When reading `git status --short`, check COLUMN 1 only.
