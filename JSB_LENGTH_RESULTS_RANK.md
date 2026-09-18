# READING -- the rank account of the MapPoPE collapse is REFUTED

**A1 NOT CONFIRMED.** MapPoPE's far-bucket NLL is 4.078 at r=1, 4.616 at r=2, 4.630 at r=4. The r=1
direction is right (-0.538, 5/5 seeds) but it is UNMEASURED (MDE 0.955) and, more to the point, far
too small to matter: at every rank MapPoPE is 4.1-4.6 where MapWM at the same rank is 0.91-1.66.
Lowering the rank does not rescue it; r=4 does not worsen it (+0.015, 1/5).

**A2 fires against the account.** The same rank effect appears on the MapWM control row and is
DETECTABLE there: r1 - r2 = -0.485 at 2-4x (5/5, MDE 0.115), growing with length (5/5, detectable).
So lower rank helps length extrapolation ON BOTH ROWS by a similar amount (-0.538 vs -0.485). That is
a general property of the rank bottleneck, not something about PoPE's frequencies -- which is exactly
the disconfirming pattern registered in A2.

**A3: the rank account is refuted and two suspects remain**, neither reachable by this knob:
the FREQUENCY COUNT (PoPE carries one phase per element, 32 per head here, against RoPE-style
rotation's 16 per head) and the ANGLE MAGNITUDE (the omega base, untouched in all of these runs).
Testing the first needs a PoPE variant with pair-wise frequencies, which is not built. Testing the
second is an omega-base sweep, which is a constructor argument and therefore cheap but not eval-only.

## The finding that came out sideways: lower rank is better for extrapolation here

**MapWM at r=1 is the best arm in the batch at every bucket** -- 0.5270 / 0.6671 / 0.9115, against
MapWM r=2's 0.5382 / 0.7835 / 1.3969 and RoPE's 0.5512 / 0.8116 / 2.0587. The r1 - r2 gap is
detectable in distribution (-0.0112, 5/5) and grows to -0.485 at 2-4x.

This runs against this project's standing "use r=4 on the MapWM family" recommendation, which was
measured on the torus navigation task (RANK_SWEEP.md: r=4 buys +0.085 at 8x training length there).
Here the ordering is r1 > r2 > r4 for extrapolation and r=4 is the worst path-integrated arm
(1.660 at 2-4x). Both cannot be a general rule about rank. The difference between the tasks is what
the increment has to encode: on the torus it is a 2-D displacement that needs a well-conditioned
basis, while on serialised music there is no displacement to represent and a wider bottleneck mostly
lets more content drive the phase. Recorded as a conflict, not resolved -- the torus result stands on
its own task and this one on this task.

