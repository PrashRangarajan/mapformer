---
name: em-vs-wm-mechanism
description: MapWM is NOT additive (per-pair content-set kernel); EM's kernel is shared. EM's recency deficit is SEARCH -- the solution exists and can be held, but is never found.
metadata:
  type: feedback
---

**The old "AND-gate vs OR-gate" story is wrong on the WM side.** Verified in `model.py:224-232`:

- **MapEM**: `softmax(A_X (*) A_P)`, with `A_P = q0^T R(dtheta) k0`. The position kernel is the
  SAME for every query-key pair (content-independent).
- **MapWM**: `Q_t^T R(theta_s - theta_t) K_s = sum_b |Q_b||K_b| cos(omega_b dS + phi_b(q,k))`.
  Query and key content set both the amplitudes and the phases of its kernel. Not additive;
  no OR-gate.

**Why:** the additive framing came from a 2026-05-10 summary. Nobody checked it against the
code, and for four months it propagated into CLAUDE.md, THEORY_KERNEL.md's Thm 3 and
TALE_OF_TWO_ALGORITHMS.md. An adversarial audit caught it (AUDIT_2026-09-10.md).

**How to apply:**
- Never describe WM as additive. The contrast is a shared kernel (EM) vs a per-pair kernel (WM).
- The only EM/WM difference measured is recency (k-back): single-`p0` EM - WM = -0.375 (0/8).
  Map tasks tie within 0.004. Do not call it a function-class limit:
  - **exists**: a query token whose Delta rewinds the count makes the retrieval offset zero.
    With that rewind installed and frozen, EM scores 1.000 on 8/8 seeds.
  - **largely holdable**: installed trainable at 8x weight scale, EM keeps 0.941. At 1/64 scale
    Adam erodes the code (0.642). That erosion was ~84% of the gap once read as "the landscape
    rejects the solution".
  - **never found**: 0/40 from-scratch EM runs learn a rewind.
  So it is SEARCH. The open question is why search fails.
- **Phase freedom** (letting `k0`'s per-block phases move) is +0.146 vs a matched-optimiser
  control (22/24; +0.113 on fresh seeds). It helps without a rewind; the mechanism is
  unidentified. Magnitude freedom and initial coherence are null.
- The early-window account (a random content gate dismantles the position code) is refuted: a
  late release is dismantled within 1-6 epochs.
- Untested hypothesis only: the recency ordering WM > EM-sep > EM-P0 tracks phase freedom
  (per-pair > n_b global > none).
- The 2026-05-10 regime table was already retracted; do not revive it.

Full account: `EM_WM_STATE.md`. See [[project-clock-vs-map]], [[feedback-existence-before-mechanism]].
- Settled 2026-09-11 (NOLEAK_RESULTS.md): installed at 8x scale with the content->Delta leak closed, trainable
  EM holds the recency rewind at 1.000 (8/8). The deficit is search, not representation or stability.
