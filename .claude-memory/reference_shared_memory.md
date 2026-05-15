---
name: Memory is shared across chats via git
description: This memory directory lives in the repo at .claude-memory/. The standard ~/.claude/projects/.../memory/ path is a symlink. Pull before reading, push after writing.
type: reference
---

This memory directory has been migrated into the git repo at `<repo>/.claude-memory/`. The standard Claude Code path
`~/.claude/projects/-home-<user>-<repo>/memory/` is a **symlink** to it on each machine, so the auto-memory tooling reads/writes the repo files transparently.

**Why:** The user runs more than one chat session against this repo (possibly on different machines). Having memory tracked in git makes it cross-session-shared, not per-machine.

**How to apply:**
- Treat memory like any other tracked artifact: `git pull` before relying on it, `git add .claude-memory/` + commit + push after writing.
- If two chats might be active at once, push memory updates promptly so the other chat sees them on its next pull.
- On a new machine, run `bash .claude-memory/setup_symlink.sh` once to create the symlink. After that, normal Claude Code memory writes go to the repo automatically.
- If you encounter a `MEMORY.md` that disagrees with the repo's `RESULTS_PAPER.md` or `CLAUDE.md`, trust the latter — they have stricter freshness guarantees.

**Caveats:**
- No concurrency control. If two chats edit the same memory file simultaneously, last writer wins. Mitigate by keeping memory edits small and pushing immediately.
- The symlink is local to each machine — it's not in git. The bootstrap script `.claude-memory/setup_symlink.sh` handles initial setup per machine.
