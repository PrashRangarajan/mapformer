---
name: verify-before-relaying
description: Agents, web summarisers and my own probes all returned confident wrong answers this session. Verify the load-bearing claim yourself before relaying or acting on it.
metadata:
  type: feedback
---

Four distinct intermediaries produced confident wrong output in one session, and
in every case a direct check settled it in minutes.

- **A WebFetch summariser** reported GRAPE as "purely index-driven, no
  content-dependence". The PDF text contains content-gated path-integral forms and
  a cumulative phase `Phi_t = sum omega_l` of per-token frequencies. **Always
  pdftotext the saved PDF and grep it yourself** rather than trusting the prose
  summary; WebFetch saves the binary locally and names the path.
- **Two review agents** were largely right but both mis-stated details, and one
  claimed a paper table (MapFormer v4 r=1) that is absent from the locally held
  v3. Relay only what you verified; mark the rest explicitly unverified.
- **My own probes**, three times: a test that varied two coordinates at once so
  nothing could read as structured; a test of Toeplitz structure, which is the
  wrong invariant once increments are content-dependent; and a perturbation that
  was purely REAL, so a test of the PHASE never fired and both arms looked
  identical.
- **A loss-matched analysis** that regressed accuracy on its own eval NLL --
  circular by construction, since both are readouts of the same softmax. It
  appeared to null every effect in the batch including an established one. The
  training loss lives in the checkpoint's `losses`, not in the eval JSON's third
  field.

**Why:** the failure mode is identical each time -- the output is well-formed and
plausible, so nothing signals the error. Only an independent check does.

**How to apply:** before relaying a claim that would change what you do, ask what
one-command check would falsify it, and run that. For a paper, read the text. For
a probe, verify what it measures on a case whose answer you already know. For a
correlation, check the two variables are not the same quantity twice.
