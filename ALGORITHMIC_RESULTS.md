# Path integration on the looped literature's own tasks

Trained at L=16, evaluated to L=256. 5 arms x 2 tasks x 8 seeds, one
batch. Gated before training: every trivial baseline sits at chance at
every length, worst excess +0.0122 (`ALGORITHMIC_GATES.md`).

## parity (chance 0.5000)

| arm | params | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|---|
| RoPE <br><sub>index, flat</sub> | 199,042 | 0.655 | 0.573 | 0.536 | 0.517 | 0.509 |
| Vanilla <br><sub>path-int, flat</sub> | 199,490 | 0.971 | 0.899 | 0.703 | 0.600 | 0.550 |
| RoPELooped <br><sub>index, loop x4</sub> | 199,042 | 0.986 | 0.812 | 0.655 | 0.576 | 0.537 |
| Looped <br><sub>path-int, loop x4</sub> | 199,490 | 0.988 | 0.989 | 0.850 | 0.662 | 0.584 |
| LoopedSampled <br><sub>path-int, loop SAMPLED</sub> | 199,490 | 0.989 | 0.989 | 0.876 | 0.700 | 0.600 |

## copy (chance 0.1250)

| arm | params | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|---|
| RoPE <br><sub>index, flat</sub> | 199,042 | 1.000 | 0.126 | 0.126 | 0.126 | 0.125 |
| Vanilla <br><sub>path-int, flat</sub> | 199,490 | 1.000 | 0.138 | 0.132 | 0.126 | 0.127 |
| RoPELooped <br><sub>index, loop x4</sub> | 199,042 | 1.000 | 0.130 | 0.124 | 0.125 | 0.125 |
| Looped <br><sub>path-int, loop x4</sub> | 199,490 | 1.000 | 0.127 | 0.127 | 0.131 | 0.185 |
| LoopedSampled <br><sub>path-int, loop SAMPLED</sub> | 199,490 | 1.000 | 0.129 | 0.131 | 0.186 | 0.127 |

## H1 -- does path integration help PARITY more than COPY?

| length | parity: path-int - index | copy: path-int - index | difference |
|---|---|---|---|
| L=16 | +0.316 (sd 0.167, t +5.36, MDE 0.165, 8/8) | +0.000 (sd 0.000, t +0.00, MDE 0.000, 0/8) | +0.316 |
| L=32 | +0.326 (sd 0.079, t +11.73, MDE 0.078, 8/8) | +0.013 (sd 0.030, t +1.20, MDE 0.030, 5/8) | +0.314 |
| L=64 | +0.167 (sd 0.037, t +12.88, MDE 0.036, 8/8) | +0.006 (sd 0.013, t +1.35, MDE 0.013, 6/8) | +0.161 |
| L=128 | +0.083 (sd 0.020, t +11.88, MDE 0.020, 8/8) | -0.000 (sd 0.002, t -0.11, MDE 0.002, 5/8) | +0.083 |
| L=256 | +0.041 (sd 0.011, t +10.16, MDE 0.011, 8/8) | +0.002 (sd 0.004, t +1.70, MDE 0.004, 6/8) | +0.038 |

## H2 -- does the loop improve LENGTH GENERALIZATION?

Retention = accuracy above chance at L, as a fraction of that at L=16.

| task | arm | L=16 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|---|
| parity | RoPE | 1.00 | 0.47 | 0.24 | 0.11 | 0.06 |
| parity | Vanilla | 1.00 | 0.85 | 0.43 | 0.21 | 0.11 |
| parity | RoPELooped | 1.00 | 0.64 | 0.32 | 0.16 | 0.08 |
| parity | Looped | 1.00 | 1.00 | 0.72 | 0.33 | 0.17 |
| parity | LoopedSampled | 1.00 | 1.00 | 0.77 | 0.41 | 0.20 |
| copy | RoPE | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| copy | Vanilla | 1.00 | 0.02 | 0.01 | 0.00 | 0.00 |
| copy | RoPELooped | 1.00 | 0.01 | -0.00 | 0.00 | 0.00 |
| copy | Looped | 1.00 | 0.00 | 0.00 | 0.01 | 0.07 |
| copy | LoopedSampled | 1.00 | 0.00 | 0.01 | 0.07 | 0.00 |

## H3 -- does SAMPLING the loop count beat a fixed count at long L?

| task | length | LoopedSampled - Looped | verdict |
|---|---|---|---|
| parity | L=64 | +0.027 (sd 0.110, t +0.69, MDE 0.109, 4/8) | UNMEASURED |
| parity | L=128 | +0.038 (sd 0.148, t +0.73, MDE 0.147, 4/8) | UNMEASURED |
| parity | L=256 | +0.015 (sd 0.066, t +0.65, MDE 0.066, 5/8) | UNMEASURED |
| copy | L=64 | +0.005 (sd 0.015, t +0.86, MDE 0.015, 4/8) | UNMEASURED |
| copy | L=128 | +0.055 (sd 0.117, t +1.32, MDE 0.116, 4/8) | UNMEASURED |
| copy | L=256 | -0.058 (sd 0.162, t -1.02, MDE 0.160, 5/8) | UNMEASURED |

## Verdict

**H1 -- CONFIRMED on parity, but the CONTROL FAILED.**

Path integration minus index on parity is **+0.316 / +0.326 / +0.167 / +0.083 /
+0.041** at L=16/32/64/128/256, **8/8 seeds at every length**, t = 5.4 to 12.9. The
mechanistic prediction holds: theta = omega * cumsum(Delta(x_t)) wrapped mod 2*pi is
a natural parity register, and the model finds it. This is the largest, most
consistent effect this project has measured outside navigation.

**But copy cannot do the job it was included for.** It has NO dynamic range at any
length: every arm scores exactly 1.000 at L=16 (ceiling) and 0.120-0.211 at L>=32
against a chance of 0.125 (floor). "Path integration does not help copy" is
therefore uninformative -- nothing helps copy, because no arm generalises on it at
all. The pipeline-artifact worry that copy was meant to rule out is NOT ruled out.
A proper control needs a task with dynamic range where path integration should not
help; copy at these lengths is not one.

**H2 -- the loop improves length retention in BOTH rows, ADDITIVELY.**

Retention at L=128: index 0.11, path-int 0.21, index+loop 0.16, path-int+loop 0.33,
sampled 0.41. The retention table makes that look super-additive, but it is a
normalisation artifact -- dividing by different baselines. On the RAW accuracy
scale the 2x2 interaction is **+0.028 / +0.003 / +0.007** at L=64/128/256, all
inside their MDEs, and the loop's main effect is nearly identical in the two rows
(L=128: +0.062 path-int, +0.058 index; L=64: +0.146 and +0.118).

**So the loop and path integration are ADDITIVE here.** That directly contrasts
with Match-Query, where the same 2x2 gave an interaction of +0.315. The
super-additivity is Match-Query-specific and does NOT generalise to the
literature's own task. Read retention ratios with suspicion: they normalise by a
quantity that differs across arms.

(At L=32 the interaction is -0.150 and detectable, but that is ceiling compression
-- path-int sits at 0.899 and path-int+loop at 0.989, with no room left.)

**H3 -- UNMEASURED.** Sampling the loop count is directionally positive on parity
at every long length (+0.027 / +0.038 / +0.015) but never clears its MDE, at 4-5/8
seeds. Our torus finding that adaptivity rescues length generalization does NOT
replicate here at n=8.

## Scope

Two tasks, one train length, one loop count for the fixed arm, n=8. Binary addition -- the third task in arXiv 2409.15647 -- is not included; its formatting is fiddly and two clean tasks beat three sloppy ones. These models are 1 layer at d=128, far smaller than the literature's, so absolute numbers are not comparable to theirs; the CONTRASTS are what this run is for.
