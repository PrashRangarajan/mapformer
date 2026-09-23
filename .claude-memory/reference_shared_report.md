---
name: reference-shared-report
description: The user's shared language-results report -- live artifact URL, repo source file, and the republish-to-same-URL rule.
metadata:
  type: reference
---

**Live link (shared by the user):** https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc -- "Position Codes on
Language", version 5 as of 2026-09-23, shared with anyone with the link (viewers see updates immediately).

**Source of truth:** `report/language_summary.html` in the repo. It was built in a session scratchpad
(`language_results.html`, ephemeral) and copied into the repo byte-identical to what was published.

**How to apply:**
- To change the report: edit `report/language_summary.html`, then publish with the Artifact tool passing
  `url=https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc`. A publish without `url` creates a NEW link and the
  one the user has already shared goes stale. In a new session, `action: "read"` the URL first (the tool
  refuses a publish to an artifact the session has not read), then build on the version it returns.
- Commit the edited HTML after every republish so the repo copy never lags the live page.
- Structure (user's stated preference): short version first, positive results first, failures briefly at
  the end, every term defined. Content must follow CLAUDE.md's newest blocks -- e.g. never cite the
  retracted code OOD numbers.
