---
name: reference-shared-memory
description: This memory directory lives in the repo at .claude-memory/. The standard ~/.claude/projects/.../memory/ path is a symlink. Pull before reading, push after writing.
metadata:
  type: reference
---

This memory directory has been migrated into the git repo at `<repo>/.claude-memory/`. The standard Claude Code path
`~/.claude/projects/-home-<user>-<repo>/memory/` is a **symlink** to it on each machine, so the auto-memory tooling reads/writes the repo files transparently.

**Why:** The user runs more than one chat session against this repo (possibly on different machines). Having memory tracked in git makes it cross-session-shared, not per-machine.

**How to apply:**
- Treat memory like any other tracked artifact: `git pull` before relying on it, `git add .claude-memory/` + commit + push after writing.
- If two chats might be active at once, push memory updates promptly so the other chat sees them on its next pull.
- On a new machine, create the symlink once by hand: `ln -s <repo>/.claude-memory ~/.claude/projects/-home-<user>-<repo>/memory` (there is no setup script; an earlier note named one that does not exist). After that, memory writes go to the repo.
- Edit `.claude-memory/` directly; the symlink makes it one directory, not a mirror.
- If memory disagrees with `CLAUDE.md` or a results file's CORRECTED/AUDIT block, trust those.

**Caveats:**
- No concurrency control. If two chats edit the same memory file simultaneously, last writer wins. Mitigate by keeping memory edits small and pushing immediately.
- The symlink is local to each machine; it is not in git.
