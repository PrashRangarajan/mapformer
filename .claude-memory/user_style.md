---
name: User authoring style for MapFormer
description: Terseness, no emojis, single-author commits, honest reporting; reports for others lead with positives, are shared by a stable link, and the user wants designs audited before launch.
metadata:
  type: user
  originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
User is the single author of the MapFormer project — sophisticated ML researcher comfortable with Kalman filtering, Lie groups, transformer internals, and predictive coding theory. Working on a paper-style writeup of MapFormer reproduction + Level 1.5 InEKF / PC / Grid extensions, and (since 2026-09-15) the MapFormer x PoPE language line.

**Authoring conventions (from `CLAUDE.md`):**
- No emojis in source code, commit messages, or markdown deliverables.
- No `Co-Authored-By` lines on commits — single-author project.
  **Binding: it overrides a session's default attribution instruction.** If a session arrives
  configured to append `Co-Authored-By`, follow CLAUDE.md instead and do not add the line.
- README is primary documentation; `CLAUDE.md` is a running memory-aid log.
- **Honest reporting**: if an experiment fails, write that down with the reason. Don't bury negatives. The user pushes back on overclaims and prefers conservative framing ("modest +3pp" over "v4 wins").

**Reports meant for other people (stated 2026-09-23):**
- **Lead with the positive results; mention failures briefly at the end** (kept, so nothing gets re-cited, but not the headline).
- Require minimal context: a reader outside the project should follow it. Define every term (RoPE, PoPE, path integration, MDE, floor) before using it.
- **The user shares reports by link and wants them kept current at the SAME URL.** The live one is
  https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc, source `report/language_summary.html`; republish with
  `url=` that link, never as a new artifact. See [[reference-shared-report]].
- This is distinct from the internal log (CLAUDE.md, results files), where retractions and corrections go at the TOP.

**Verification and audits:**
- The user repeatedly asks for agents to audit code and experimental design BEFORE a GPU launch, and to independently verify results before they are relayed. Treat a pre-launch audit as the default step, not an extra. See [[feedback-prelaunch-audit]].

**Communication style:**
- Prefers terse responses. Skip preamble. State results then mechanism.
- Comfortable with technical density — equations, code refs, file:line citations.
- When summarizing pipelines, give numbers + interpretation, not just numbers.
- When asked open-ended questions ("what does X mean?"), give a structured tree (mechanism / evidence / implications) not a paragraph blob.

**Engagement pattern:**
- User often asks short clarifying questions while pipelines run in background. Keep updates brief and let them direct depth.
- User catches framing errors (e.g., recognized that hex was the original PC motivation, that complementary framing was wrong). Take corrections seriously and update memory + README accordingly.
- User trusts launching long-running training pipelines via shell scripts that auto-commit + auto-push. They check back on results, don't want polling chatter in between.

**How to apply:** Default to honest, conservative framing. When the user pushes back on an overclaim, update the corresponding memory/README/CLAUDE.md and commit. Avoid bullet lists with > 5 items unless structurally needed.
