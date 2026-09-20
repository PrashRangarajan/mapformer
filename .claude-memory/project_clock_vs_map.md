---
name: project-clock-vs-map
description: Cancellation chooses whether the accumulator is a map or a clock. Explains sign and rank; recency does NOT need a clock (a rewind solves it); forget-gate account untested.
metadata:
  type: project
---

Attention sees only the interval sum `θ_s − θ_t = ω·Σ_{t<u≤s} Δ_u`. What that
quantity **is** depends entirely on whether increments cancel:

| | Δ signed | Δ monotone |
|---|---|---|
| the interval sum is | **net displacement** | **path length** |
| two routes A→B | agree | disagree |
| injective in | space, blind to time | time, blind to place |
| the accumulator is | a **map** | a **clock** |

Mutually exclusive by construction, and **each is correct for one job**. A cognitive
map NEEDS same-place-different-time to collide — that IS revisit prediction. Language
needs the opposite. So CARoPE/GRAPE-AP/CoPE being monotone is not an oversight; on
text a monotone clock is the right object, which is why RoPE is the default there and
why nothing went visibly wrong.

**Do not say a signed increment is simply better.** That framing was in the documents
and is wrong. **Nor say recency needs a clock** -- see the scope section at the end.

## The growth exponent is a readout of which question, not a score

Measured (`LOCALISATION.md`, eval-only). All arms start from the same accumulator
range at training length (88–94) and separate by exponent, `range(S) ~ T^α`:

| arm | α | opposition |
|---|---|---|
| signed r=4 | **0.518** | 0.125 |
| signed r=4 (rank sweep) | 0.524 | 0.092 |
| signed r=2 | 0.619 | 0.495 |
| monotone r=4 | **0.943** | 1.849 |

α≈0.5 = diffusive random walk (measuring position); α≈0.94 = ballistic (measuring
elapsed time). `r(opposition, α) = +0.9995` across two independent batches. **One
mechanism covers the sign result and the rank result.** The ideal for a bounded map
would be α=0 and nothing achieves it — everything accumulates without wrapping.

## Scope: it explains TWO of four, not "almost everything"

`ACCUMULATOR.md`: neither the forget gate (−0.051, MDE 0.092) nor PoPE (+0.006, MDE
0.177) nor Level15 (+0.009, MDE 0.191) changes α. All unmeasured.

**And the positive control FAILED.** "The InEKF's wrap bounds the accumulator" is
FALSE — it wraps the INNOVATION, so `θ̂ = θ_path + bounded correction` and
`range(θ̂) = 285.6` vs `range(θ_path) = 283.9` at T=1024. Asserted twice without
measuring. Withdrawn.

## Candidate account of the forget gate (consistent, untested)

A signed phase cannot represent *how long ago*. The forget gate adds a second,
content-dependent, **monotone** accumulator (`Σ log γ`, measured α = **+0.956**),
restoring the clock. Fits everything: needs a LIVE λ (frozen → baseline, −0.016) but
not DECAY; 5/8 seeds learn λ<0 and gain most (r = −0.516); negative λ is still
monotone, counting up. **It was never a forget gate — it's a clock.**
Predictions, untested: any monotone content-dependent increment of either sign should
reproduce the gain; a content-INDEPENDENT constant should reproduce much of it; the
gain should vanish where there is no recency structure.

See [[project-sign-axis]], [[reference-positional-landscape]].

## alpha is a RE-DESCRIPTION, not a third cause (added 2026-09-08)

Two corrections, both from the user asking what alpha is actually for.

**It is nearly collinear with the opposition score: r = +0.9995.** Opposition is
read straight off the learned action code, is simpler and more interpretable, and
alpha carries almost no information it does not. Nothing rests on alpha that could
not have rested on opposition. Its contribution is **economy** — it makes the sign
result and the rank result one finding at two severities rather than two.

**And "vary alpha deliberately and check that degradation follows" is MALFORMED.**
alpha is not a parameter of any model here; it is a statistic fitted to a trained
one. Both ways to move it fail: interpolating signed -> monotone moves alpha but
destroys the map in the same stroke, and bounding the accumulator is either a NO-OP
(theta enters through cos/sin, so theta and theta+2pi are already identical) or it
breaks additivity and with it interval-relativity. **Within this frame alpha may not
be independently controllable at all** — it is set by the cancellation, whose only
levers are the sign and rank axes.

**The prior question, which I had skipped:** why should a growing accumulator cost
anything when the code is periodic? The standard answer — low-frequency channels at
untrained phases — was imported here and **refuted** (ablating them costs MORE at
OOD). So alpha's association with length-degradation is established and its **route
is not**.

**One narrow use survives**: opposition needs labelled opposite actions and cannot
be computed on text at all; alpha needs only trajectories. It is the form this
measurement takes in the setting the literature actually works in. Not yet measured
there — the language trainer saves no checkpoints.


## Scope, corrected by the 2026-09-10 audit: recency does NOT need a clock

"Mutually exclusive" holds only for a scalar or fully constrained accumulator. The
model's is rank-r per head, so one subspace can cancel while another counts
(THEORY_KERNEL.md Thm 1, now scoped). And a clock is not the only way to do k-back.
A **signed** accumulator solves it if the query token `q_k` carries Delta = -(k-1),
which rewinds the count so the retrieval offset is zero for every query. A single-`p0`
EM kernel then selects the answer exactly (1423/1423; without the rewind 0.0857). The
full EM model with that rewind installed and frozen scores 1.000 on 8/8 seeds
(WARM_RESULTS.md).

So the crossover's recency half says a monotone increment is **harmless** there
(-0.004), not that recency **needs** one. The map half (-0.280 torus, 12/12) is
unaffected. From scratch, EM finds it per query token, wrapped, for ~half the k (the "0/40" was a linear readout; SEARCH_RESULTS.md). That is a search
problem, and the frame says nothing about it -- see [[em-vs-wm-mechanism]] and
`EM_WM_STATE.md`.

**Numbers to keep (restored after the 2026-09-11 consolidation dropped them):** growth exponent alpha of
the UNCONSTRAINED arm is 0.591 on the torus and 0.967 on recency (se 0.009), constrained arms pinned
at ~1.0 on both (`RECENCY_RESULTS.md`). Forcing monotone costs -0.280 on the torus, -0.004 on recency.

**NEW CONSEQUENCE (2026-09-18, `THEORY_MAPPOPE.md`, `T1_RESULTS.md`): PoPE-style decoupling is only
safe on the MAP side.** PoPE removes the pairwise phase (its `delta_c` is a per-head constant and its
amplitudes are non-negative), so its kernel is calibrated absolutely in `S_t - S_s` and content cannot
compensate when that argument leaves the trained range. On a clock (music, alpha = 1.00) MapPoPE
collapses beyond the training context (4.62 vs MapWM 1.40); on a bounded accumulator (Dyck-2, where
opens and closes cancel) it is the best arm out of distribution. Centring the increment -- 3x smaller
excursion -- recovers it to 0.911, a gain 5.7x larger than the same treatment gives MapWM, at a
detectable in-distribution cost.

**BOUNDARY CONFIRMED ON THREE TASKS (2026-09-19, `T3GEN_RESULTS.md`).** A per-token phase added to
PoPE pays only where the accumulator is a CLOCK: Bach with a 512 context (4.616 -> 0.616 at 2-4x with
a forced init) and nothing where it is bounded -- Dyck-2 -0.046 against its own inert twin
(detectable), torus -0.021 at 8x with 9x the seed spread. Zero-initialising those phase heads is a
BAD PRIOR: forcing the init to 0.1 improves both in-distribution and extrapolation, detectably.

**BOUNDARY CONFIRMED ON THREE TASKS (2026-09-19, `T3GEN_RESULTS.md`).** A per-token phase added to
PoPE pays only where the accumulator is a CLOCK: Bach with a 512 context (4.616 -> 0.616 at 2-4x with
a forced init) and nothing where it is bounded -- Dyck-2 -0.046 against its own inert twin
(detectable), torus -0.021 at 8x with 9x the seed spread. Zero-initialising those phase heads is a
BAD PRIOR: forcing the init to 0.1 improves both in-distribution and extrapolation, detectably.
Reruns at the corrected init (0.1) confirm it as a DOUBLE DISSOCIATION: Dyck -0.047 vs its inert twin
(unchanged from the zero-init verdict) and torus 0.911 vs 0.963 at l=2048, i.e. forcing the phase
HURTS on bounded accumulators while it is the best configuration on a clock. The model does NOT
decline the phase where it is useless (Dyck keeps 0.825 rad, more than Bach's zero-init 0.412) -- it
uses it and is worse for it.
Reruns at the corrected init (0.1) confirm it as a DOUBLE DISSOCIATION: Dyck -0.047 vs its inert twin
(unchanged from the zero-init verdict) and torus 0.911 vs 0.963 at l=2048, i.e. forcing the phase
HURTS on bounded accumulators while it is the best configuration on a clock. The model does NOT
decline the phase where it is useless (Dyck keeps 0.825 rad, more than Bach's zero-init 0.412) -- it
uses it and is worse for it.

**A DECAY ENVELOPE IS A PROXIMITY PRIOR IN THE POSITION VARIABLE'S OWN METRIC (2026-09-19,
`DYCK_DECAY_RESULTS.md`).** The same ALiBi-style envelope (48 params) drives index PoPE to the no-stack
n-gram level beyond distance 8 on Dyck (0.499 at d 9-32) and IMPROVES path-integrated MapPoPE there
(0.778 at d 33+, best Dyck arm at F1 0.956). Reason, measured: the accumulated distance |S_t - S_s|
correlates 0.755 with stack-DEPTH difference and only 0.278 with token distance (8 seeds,
`probe_dyck_metric.py`); the crossed arm that would isolate the metric is NOT run, so decaying over it
means "prefer keys at similar depth". On a clock accumulator (Bach) it degenerates to recency, which is
why decay and path integration looked interchangeable there.

**T2 FAILS (2026-09-19, `T2_RESULTS.md`) -- the within-task test does not reproduce the boundary.**
Forcing monotone increments on Dyck turns its accumulator from a map into a clock (alpha 0.58/0.62 ->
1.06/1.04, range 4.2x instead of 2.3x) and PoPE's encoding then helps MORE, not less (+0.159 against
+0.058 signed; difference of differences +0.100 where negative was predicted, unmeasured). So the
clock/map boundary is NOT established within a task; the three-task pattern is confounded with model
size, sequence length and excursion SIZE (Bach reaches range 551 at 4x, monotone Dyck only 24).
The Bach interventions (T1, T3) stand; the generalisation from them does not.

**T2 FAILS (2026-09-19, `T2_RESULTS.md`) -- the within-task test does not reproduce the boundary.**
Forcing monotone increments on Dyck turns its accumulator from a map into a clock (alpha 0.58/0.62 ->
1.06/1.04, range 4.2x instead of 2.3x) and PoPE's encoding then helps MORE, not less (+0.159 against
+0.058 signed; difference of differences +0.100 where negative was predicted, unmeasured). So the
clock/map boundary is NOT established within a task; the three-task pattern is confounded with model
size, sequence length and excursion SIZE (Bach reaches range 551 at 4x, monotone Dyck only 24).
The Bach interventions (T1, T3) stand; the generalisation from them does not.

