---
name: PC and Kalman are duals, not complements
description: The earlier "forward + inverse model are complementary" framing is wrong. Use the duality framing in any new writeup.
type: feedback
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
Earlier README/CLAUDE.md sections claimed PC (forward model `g(θ)→ô`) and InEKF (inverse model `h(o)→θ`) were complementary corrections. **This framing is wrong** and was corrected in the 2026-04-29 session.

**The right framing:** PC's forward map and InEKF's measurement model are mathematical duals — same Bayesian posterior over θ written from opposite sides. When both operate on the same θ̂ with the same inputs, they target the same fixed point ("θ̂ such that obs and predicted obs are consistent under some learned map"). Gradient descent finds the trivial joint minimum: `g ∘ h ≈ identity`, achieved by `R → 0` so `θ̂ ≈ h(x_t)` (the autoencoder bypass empirically observed in Level15PC at log_R ≈ -3, near the -5 clamp).

**Why:** The aux_coef sweep showed monotone dose-response — more PC → worse lm200 OOD T=512 (0.0→0.79, 0.1→0.72, 0.3→0.55). Stop-gradient fixes (NoBypass v2, v3 tighter clamp) closed the direct route but PC still leaked via shared `action_to_lie`, blowing |θ̂| up to ~3840 at T=512 (LENGTH_DIAGNOSTIC.md). Only v4 (full PC isolation: detach both θ̂ and target embedding) avoids the collapse — but at that point PC has zero gradient flow into the main model.

**How to apply:** When discussing PC + Kalman in any new document, paper section, or response:
- Don't say "complementary" or "stack" them.
- Frame as duals; coupling them creates a degenerate optimum gradient descent will find.
- v4's modest +3pp lm200 win cannot mechanistically come from PC (gradient detached); attribute to RNG drift or optimizer-state side effects, not PC representation pressure.
- Old section in README "Predictive Coding (forward model) and InEKF (inverse model) are complementary" was rewritten as "Predictive Coding and InEKF are duals, not complements (revised)" — keep that framing.
