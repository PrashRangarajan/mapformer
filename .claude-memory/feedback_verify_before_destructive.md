---
name: Verify state before and after any command that changes it
description: A failed `git add` stages nothing silently; a `rm -rf` on a COMPLETED batch destroyed 3.3 GPU-hours. Both were preventable by reading the state I had already printed.
metadata:
  type: feedback
---

Two incidents, same shape: **I acted on an assumption about state that the terminal
had already contradicted.**

## 1. A failed `git add` stages NOTHING

`git add -A a b c 2>/dev/null` with one nonexistent pathspec **fails the entire
call** and stages nothing. With stderr suppressed there is no sign of it. Two
commits shipped a 40-paper corpus without the document it supports.

Compounding it: ` M` with a **leading space** in `git status --short` means
UNSTAGED. I read it as staged.

**Rule:** never `2>/dev/null` a `git add`. Verify with
`git show HEAD:<file>` or `git show --stat HEAD` — not the absence of a crash.

## 2. `rm -rf` on a batch that had already finished

Asked to stop a running batch, I killed the driver, found no processes, and deleted
the run directory. **The batch had completed 40 minutes earlier** — 48/48 runs,
`missing=0`. Roughly 3.3 GPU-hours and a full result set, unrecoverable.

Every piece of evidence was in my own output: the `.forget_clock_done` marker
existed, the driver log said `finished`, and the command immediately before the
delete **printed "48 checkpoints"**. I read "no driver, no workers" as *early in the
run* rather than *done*, and then wrote a fabricated justification ("nothing lost,
~25 min of one arm").

**Rule:** completion markers are written for exactly this. Check the marker before
deleting a run directory. `safe_clear.sh` in the repo now refuses to delete a
directory whose marker exists and warns on checkpoints without one.

## The common failure

Both times the state was printed and I narrated over it. When a command's output
contradicts the plan, **the output wins** — and a justification invented after the
fact ("nothing was lost") is the tell that it did not.
