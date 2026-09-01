---
name: reference-language-and-pope
description: enwik8 numbers with power caveats; what theta means without actions; PoPE modifies magnitude not angle.
metadata:
  type: reference
---

**enwik8, flat 9-layer, ~28.6M params, 36k iters, seq 512** (`enwik8_long/*.json`):
MapPoPE-Flat 1.3740 (n=3) · PoPE-Flat 1.3746 (n=1) · Vanilla/MapWM 1.3758 (n=1) ·
RoPE 1.3799 (n=3). Only MapPoPE−RoPE has seeds both sides: **−0.0058, t=3.49**.

**Our −0.0041 position effect does NOT contradict the paper.** v4 §5.5 gives RoPE
19.14 vs MapWM 18.79 ppl = 0.0266 bits/BPE-token ≈ **0.0067 bits per BYTE** at 4
bytes/token. Seed sd 0.0018 → MDE at n=3 is 0.0041, the size of the effect. The
setup is UNDERPOWERED, not null. (I called it "inside the noise floor" — a rule-11
violation on my own data.) n≥5 needed.

Hint: PoPE alone −0.0053, path-int alone −0.0041, sum −0.0093, both together only
−0.0058 → on text the axes OVERLAP (sub-additive), opposite to loop×path-int on
navigation. Both singles n=1.

**theta without actions.** `ActionToLieAlgebra` reads EVERY token identically; the
model LEARNS Δ≈0 for observations (measured: actions move position 5× more). RoPE
is the **Δ≡1 special case** — `t·inv_freq` vs `cumsum(Δ)·ω`. So on text theta is a
learned content-dependent CLOCK RATE, free to be zero or negative. That is
Selective RoPE's primitive.

**PoPE's angle is NOT learned or content-dependent.** `magnitude = softplus(Q)`
comes from content; phase is `t·theta_c` with theta_c a FIXED buffer; the only
learnable angle term is `pope_delta`, a per-(head,frequency) constant. MapFormer and
PoPE modify ORTHOGONAL halves of the same polar decomposition — hence MapPoPE is
just "use both", and PoPE is NOT in the CoPE/Selective-RoPE family despite the name.

**Nothing looped has ever been run on text.** Sizing matters: resolving ~0.005 bpc
against a 0.0018 seed sd needs n≥5, so a 2×2×3 design is ~48 runs, not 18.
