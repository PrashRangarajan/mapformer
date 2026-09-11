---
name: em-vs-wm-mechanism
description: MapWM is NOT additive -- it rotates content Q,K, so its position kernel has per-pair phases; EM's kernel is content-independent. EM's recency deficit is learnability, not expressivity.
metadata:
  type: feedback
---

**The old "AND-gate vs OR-gate" story is wrong on the WM side.** Verified in `model.py:224-232`:

- **MapEM**: `softmax(A_X (*) A_P)`, with `A_P = q0^T R(dtheta) k0` -- a position kernel that is
  the SAME for every query-key pair (content-independent).
- **MapWM**: `Q_t^T R(theta_s - theta_t) K_s` = `sum_b |Q_b||K_b| cos(omega_b dS + phi_b(q,k))` --
  one kernel whose amplitudes AND phases are set by query and key content. Not additive; no OR-gate.

**Why:** the additive framing came from a 2026-05-10 summary and was never checked against the
code; it propagated into CLAUDE.md, THEORY_KERNEL.md's Thm 3 and TALE_OF_TWO_ALGORITHMS.md for four
months. An adversarial audit caught it (AUDIT_2026-09-10.md).

**How to apply:**
- Never describe WM as additive. The contrast is shared kernel (EM) vs per-pair kernel (WM).
- Before calling an EM deficit a limit of the function class, check existence: a single-`p_0` EM
  kernel solves recency exactly (1423/1423) once the query token's own Delta rewinds the count.
  EM - WM = -0.375 on recency is about TRAINING, now shown at full-model level: rewind installed and
  frozen -> 1.000 on 8/8; trainable -> dismantled early, back to scratch level; 0/40 scratch runs find it.
  The correct position code exists but is not an attractor. See WARM_RESULTS.md.
- The 2026-05-10 regime table was already retracted; do not revive it.
- Untested hypothesis only: recency ordering WM > EM-sep > EM-P0 tracks phase freedom
  (per-pair > n_b global > none). See [[project-clock-vs-map]].
