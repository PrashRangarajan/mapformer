# T2: the within-task test FAILS to reproduce the between-task boundary

Pre-registration: `T2_PREREG.md`. 8 seeds, the Dyck recipe verbatim, monotone (non-negative)
increments against the signed baselines from the same recipe.

## M (manipulation check): PASSES cleanly

Recomputed 2026-09-20 by `probe_dyck_alpha.py` (committed after an audit found these numbers had no
script and did not reproduce under another convention). The first version of this table quoted point
values with no spread; the signed arms' exponent is very noisy across seeds, so only the DIRECTION
is a stable claim:

| arm | alpha (mean +/- sd, 8 seeds) | per-seed range | range(S) L32 -> L128 |
|---|---|---|---|
| MapWM signed | 0.609 +/- 0.121 | 0.45 - 0.86 | 0.51 -> 1.28 (2.53x) |
| MapPoPE signed | 0.784 +/- 0.189 | 0.58 - 1.01 | 0.66 -> 2.39 (3.65x) |
| **MapWM monotone** | **1.017 +/- 0.004** | 1.01 - 1.03 | 7.64 -> 31.25 (4.09x) |
| **MapPoPE monotone** | **1.017 +/- 0.012** | 1.01 - 1.04 | 5.02 -> 20.59 (4.10x) |

Constraining the increment turns Dyck's accumulator from a map (alpha ~0.6, the diffusive code that
cancels) into a clock (alpha ~1.05, growing with token count), exactly as intended, within one task.

## T2 (the primary contrast): the registered prediction FAILS

F1, 8 seeds:

| arm | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| MapWM signed | 0.985 | 0.900 | 0.942 | 0.868 |
| MapPoPE signed | 0.988 | 0.923 | 0.976 | 0.927 |
| MapWM monotone | 0.957 | 0.663 | 0.888 | 0.677 |
| MapPoPE monotone | 0.961 | 0.834 | 0.929 | 0.836 |

**The registered floor readout, reported late (2026-09-20).** `T2_PREREG.md` asked whether either
monotone arm clears the no-stack n-gram floor, which is 0.904 / 0.896 / 0.903 / **0.884** for these
cells. Neither does at the primary cell: MapWM monotone 0.677 and MapPoPE monotone 0.836 are both
BELOW a stack-free lookup table, and in T2b all three arms are at or below it. On F1 most of the
residual range there is the always-legal opening brackets, so the F1 contrasts below are computed in
a region that metric cannot read. **Both verdicts were therefore re-read on the chance-anchored
metric** (closer accuracy, chance 0.500), which has real dynamic range there:

| arm | closer acc | acc at distance >= 9 |
|---|---|---|
| MapWM signed | 0.928 | 0.738 |
| MapPoPE signed | 0.978 | 0.892 |
| MapWM monotone | 0.800 | 0.561 |
| MapPoPE monotone | 0.921 | 0.737 |
| MapPoPE monotone + phase | 0.835 | 0.622 |
| MapPoPE monotone + inert twin | 0.932 | 0.765 |

T2 difference of differences: **+0.070** (MDE 0.110) over all positions and **+0.021** (MDE 0.122)
at distance >= 9 -- same sign as on F1, still where a negative was registered, still unmeasured.
T2b phase minus inert twin: **-0.097** (MDE 0.041, 0/8) and **-0.142** (MDE 0.069, 0/8) -- detectable
on both readings. **The floor problem is real and does not change either verdict.**

On F1, PoPE's encoding is worth **+0.058** on the signed (map) accumulator and **+0.159** on the monotone
(clock) one. The registered difference of differences is therefore **+0.100** where a NEGATIVE value
was predicted -- and it is unmeasured (MDE 0.244, negative on 3/8 seeds). So the claim tested is not
supported, and the observed direction is the opposite one.

Read plainly: **making Dyck's accumulator a clock does not make PoPE collapse.** It damages both arms
(monotone cannot represent push/pop, as this project measured on the torus) and PoPE's encoding helps
MORE under that damage, not less.

## What this does to the account

The boundary -- "a per-token phase pays where the accumulator is a clock, and not where it is
bounded" -- was supported by three tasks that differ in accumulator AND in dataset, model size
(6 layers / 8 heads on Bach against 1 / 1 here), sequence length and metric. T2 was the one test that
moved the accumulator alone. **It does not reproduce the pattern, so the between-task association is
not explained by the accumulator alone.**

One quantity that separates Bach from monotone Dyck, and that T2 does not control: the SIZE of the
excursion. Bach's accumulator range reaches 551 at 4x the training context; monotone Dyck's reaches
24 at 4x. If what matters is how far outside its trained band the kernel's argument goes, rather than
the growth exponent, then a clock with a small absolute excursion should behave like a map -- which
is consistent with this batch, post hoc and at one condition. That is a hypothesis this run suggests
and does not test.

**Status of the account after T2**: the two interventions on Bach (T1 centring, T3 phase) still stand
-- each manipulates one factor within one task and moves the result as the account says. What no
longer stands is the generalisation from them to a clock/map boundary ACROSS tasks: the only
within-task test of that generalisation failed, in the opposite direction.

**The obvious next test, not run**: does the per-token phase now help on monotone Dyck? The account's
second half predicts it should, since the accumulator is now a clock. If it does, the boundary
survives in the narrower form "the phase pays on clocks" even though "PoPE collapses on clocks" is
dead. If it does not, both halves are Bach-specific.

## T2b (amendment 1): the other half fails too -- the phase does NOT pay on a clock

Same task, same recipe, 8 seeds; both arms have monotone increments, so Dyck's accumulator is a
clock (alpha 1.04, verified above). They differ only in whether the per-token phase is live.

| arm | L32 D4 | L128 D12 |
|---|---|---|
| MapPoPE monotone | 0.961 | 0.836 |
| MapPoPE monotone + per-token phase | 0.961 | **0.732** |
| MapPoPE monotone + inert twin (phase gated off) | 0.968 | **0.875** |

**phase - inert twin at L128 D12 = -0.143 (MDE 0.079, 0/8 seeds positive), DETECTABLE.** The
registered prediction was positive and detectable. It is refuted with the opposite sign established,
and the damage is THREE TIMES what the same contrast showed on the signed version of this task
(-0.046). Making the accumulator a clock did not make the phase useful here; it made it more harmful.

## Verdict on the account

Both halves now fail their within-task tests, by the criteria registered before each run:

| claim | within-task test | result |
|---|---|---|
| PoPE's kernel collapses when the accumulator is a clock | T2 | FAILS: PoPE helps more (+0.159 vs +0.058) |
| a per-token phase pays when the accumulator is a clock | T2b | FAILS: phase hurts more (-0.143 vs -0.046) |

By T2b's registered falsification clause the account fails its within-task test. The precise reading,
narrower than that clause's own wording: **the boundary is not reproduced within a task at 1 layer /
1 head, and its second half is detectably reversed there.** "Bach-specific" would be a positive
localisation these data cannot support -- two failures on one task at one scale show that the
accumulator's exponent alone does not carry the effect, not that the effect is a property of Bach. What remains true is narrower and entirely within that setting: on Bach at a
512-token context, shrinking the accumulator's excursion helps MapPoPE 5.7x more than MapWM (T1),
restoring a per-token phase removes the collapse with a parameter-matched twin flat (T3), and a decay
envelope does the same for 48 parameters (DECAY). Those three interventions stand. The alpha-based
explanation that tied them to Dyck and the torus does not.

**What differs between Bach and monotone Dyck, and is now the live suspect list**: the absolute
excursion (range 551 at 4x context against 24), model size (6 layers / 8 heads against 1 / 1), the
frequency count (32 per head against 16), and how much of each task is solvable locally. None is
controlled by anything run here.

**Process note.** This is the fourth prediction of mine to fail in this line (the trade-off corollary,
the optional-freedom reading, E1 on Dyck decay, and now both halves of the boundary). The pattern is
consistent: each failure came from generalising a within-task intervention to a cross-task rule, and
each was caught only by running the within-task version. The interventions have held up every time;
the generalisations have not.

## T2c (2026-09-20): the initialisation control the audit asked for

T2b's live arm started its phase heads at 0.1 while the twin is zero-initialised with the gate off, so
"phase - inert twin" bundled the mechanism with a 0.1-scale perturbation of the starting function.
The separating arm -- gate ON, zero init -- was run: 8 seeds, same recipe.

| contrast at L128 D12 | value |
|---|---|
| phase (init 0.1) - inert twin | **-0.143** (MDE 0.079, 0/8) DETECTABLE |
| **phase (zero init) - inert twin** | **-0.063** (MDE 0.045, 0/8) DETECTABLE |
| phase (init 0.1) - plain monotone | -0.104 (MDE 0.177, 2/8) unmeasured |
| phase (zero init) - plain monotone | -0.023 (MDE 0.180, 2/8) unmeasured |
| inert twin - plain monotone | +0.039 (MDE 0.156, 5/8) unmeasured |

**The confound inflated the effect and did not create it**: with the initialisation matched, the phase
still hurts detectably on a clock accumulator, where positive was registered. Two caveats the audit
raised and this table shows: the magnitude depends on which control is used (-0.063 to -0.143), and
the "three times the signed version's -0.046" comparison in the text is against a number T3GEN itself
reports as unmeasured, so it should be read as a direction rather than a ratio.

## Batch provenance

T2's primary contrast is CROSS-BATCH: the signed baselines are `runs/dyck_bs128` (2026-09-15), the
monotone arms `runs/dyck_t2` (2026-09-19/20). The T2b and T2c arms are same-batch with their twin.
Only new classes were added to `train_dyck.py` between those dates; the environment, data seeding and
evaluation paths are unchanged.
