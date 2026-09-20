# T2: the within-task test FAILS to reproduce the between-task boundary

Pre-registration: `T2_PREREG.md`. 8 seeds, the Dyck recipe verbatim, monotone (non-negative)
increments against the signed baselines from the same recipe.

## M (manipulation check): PASSES cleanly

| arm | alpha on Dyck | range(S), L32 -> L128 |
|---|---|---|
| MapWM signed | 0.578 | 0.59 -> 1.36 (2.29x) |
| MapPoPE signed | 0.618 | 0.77 -> 2.19 (2.84x) |
| **MapWM monotone** | **1.056** | 5.78 -> 24.45 (4.23x) |
| **MapPoPE monotone** | **1.044** | 4.91 -> 21.03 (4.28x) |

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

PoPE's encoding is worth **+0.058** on the signed (map) accumulator and **+0.159** on the monotone
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
is what this batch shows. That is a hypothesis this run suggests and does not test.

**Status of the account after T2**: the two interventions on Bach (T1 centring, T3 phase) still stand
-- each manipulates one factor within one task and moves the result as the account says. What no
longer stands is the generalisation from them to a clock/map boundary ACROSS tasks: the only
within-task test of that generalisation failed, in the opposite direction.

**The obvious next test, not run**: does the per-token phase now help on monotone Dyck? The account's
second half predicts it should, since the accumulator is now a clock. If it does, the boundary
survives in the narrower form "the phase pays on clocks" even though "PoPE collapses on clocks" is
dead. If it does not, both halves are Bach-specific.
