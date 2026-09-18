# READING -- the first PoPE-dataset condition where path integration helps, and a collapse

**R2 answered, and it splits by encoding.**

- **On the RoPE row path integration wins, and the win grows with length.** MapWM - RoPE is -0.0131
  in distribution (unmeasured), -0.028 at 1-2x, and **-0.662 at 2-4x beyond the training context
  (5/5 seeds, MDE 0.335, DETECTABLE)**; the growth itself is -0.649 (5/5, DETECTABLE). MapWM is the
  best arm in the far bucket (1.397) against PoPE's 1.597 and RoPE's 2.059. This is the first
  condition on PoPE-paper data where path integration beats its index control.
- **On the PoPE row it collapses.** MapPoPE is best in distribution (0.5235, the lowest of the four)
  and then **blows up beyond the training context: 1.954 at 1-2x and 4.616 at 2-4x**, worse than
  every other arm, 0/5 seeds better, detectable at both buckets.
- **Control (eval-only, the full-context checkpoints from `runs/jsb`)**: trained at 2048 the same
  MapPoPE shows no collapse at all -- 0.540 / 0.565 / 0.612 across the three buckets, the best arm
  in every one. So the blow-up is specific to extrapolating past a training context, not a defect of
  the architecture or a bug in the buckets. (Those control numbers come from final-step checkpoints,
  which are overfit, so they are not comparable in level to the 512-trained runs -- only in shape.)
- **R3 holds on the index row and inverts on the path-integrated one**: PoPE - RoPE is -0.461 at 2-4x
  (5/5) -- PoPE's decoupling is itself a length-extrapolation win -- while MapPoPE - MapWM is +3.219.

**Mechanism, not identified.** The candidate: PoPE carries one phase per ELEMENT (64 frequencies at
d_head 64) where RoPE-style rotation carries one per PAIR (32 blocks), and path integration makes
that phase an unbounded content-driven cumulative sum. Twice as many frequencies riding on an angle
that grows without bound leaves the trained phase range sooner. Untested; it predicts that the
collapse should weaken at lower rank or with fewer PoPE frequencies, neither of which was run.

**Where this leaves the question "does path integration help on any PoPE task".** Yes, in exactly one
regime: extrapolating past the training context on the RoPE row, where it is the best arm and the
advantage grows with distance. In distribution it remains a null (Bach Chorales full-context) or a
tie (Indirect Indexing at adequate budget). And combined with PoPE it is actively harmful out of
distribution, which contradicts the Dyck-2 result where MapPoPE was the best OOD arm -- different
scale (1 layer, 1 head, sequences of 32-128) and a task with explicit push/pop structure.

