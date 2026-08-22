# Level 1.5 on compositional — the last correction-family gap

Level15 and MapWM-Flat trained in ONE batch (rule 3), n=3, n_templates=4,
50 epochs, 3 layers. Closes the last open cell in `BASELINE_TABLE.md`.

| metric | Level15 | MapWM-Flat |
|---|---|---|
| exact_acc @T=256 | 0.940 ± 0.051 | 0.919 ± 0.028 |
| **cross_nb @T=256** | **0.368 ± 0.220** | **0.260 ± 0.034** |
| cross_nll @T=256 | **0.939** | 1.385 |
| cross_nb @T=1024 | 0.180 ± 0.215 | 0.082 ± 0.009 |
| cross_nll @T=1024 | **1.419** | 2.160 |

## The accuracy mean is one outlier seed; the likelihood win is 3/3

| cross_nb @T=256 | seed 0 | seed 1 | seed 2 | paired Δ |
|---|---|---|---|---|
| Level15 | 0.240 | **0.622** | 0.241 | |
| MapWM-Flat | 0.283 | 0.221 | 0.275 | |
| Δ | −0.043 | **+0.401** | −0.034 | 2 of 3 **negative** |

Level15's ±0.220 is six times MapWM-Flat's spread. **On two of three seeds it is
worse on compositional transfer**, and the higher mean comes entirely from seed 1.
That should not be reported as an accuracy improvement.

Cross-entropy runs the other way and does so consistently — Level15 better on
**3/3 seeds at both lengths** (1.269/0.443/1.105 vs 1.365/1.499/1.290 at T=256;
1.791/0.713/1.752 vs 2.515/1.746/2.219 at T=1024).

## A reporting bug was found and fixed here

`eval_compositional.py` keyed its results by variant name
(`all_rows[v] = res`), so a 6-checkpoint run over 3 seeds x 2 variants printed
two rows containing **seed 2 only**, with no indication that four checkpoints had
been discarded. Read that way this task said "Level15 is worse on cross_nb"
(0.241 vs 0.275). The n=3 aggregate says its mean is higher. Neither reading is
the headline — the per-seed pairing is — but the file was silently reporting one
seed as if it were three. Fixed to aggregate with ± and an explicit n.

This is standing rule 8 in a new form: the retraction there was about a stale
constant, this is about a generator that quietly drops data. Both produce
confident output that is not what it claims to be.

## Where this leaves the correction family — five tasks, one pattern

| task | accuracy vs MapWM-Flat | likelihood |
|---|---|---|
| paper task, 50 ep | 1.000 vs 0.993 (ceiling) | **16x lower loss** |
| Match-Query | 0.876 vs 0.888 — no advantage | — |
| family tree | not significant (t≈0.89) | — (variance ±0.015 vs ±0.072) |
| lm200 | +0.142, but a filter-free capacity control ties it | — |
| **compositional** | **worse on 2/3 seeds** | **better 3/3, both lengths** |

Across five within-batch comparisons Level 1.5 improves **likelihood and
stability** reliably and **accuracy** almost never. That is the
"stabilisation, not inference" reading in CLAUDE.md, now measured on five tasks
rather than asserted — and it is a much narrower claim than the Kalman framing
the project was built on.
