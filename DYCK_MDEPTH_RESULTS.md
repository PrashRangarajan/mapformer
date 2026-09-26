# Dyck position effect at MATCHED nesting depth -- results

Pre-registration `DYCK_MDEPTH_PREREG.md`; gates `DYCK_MDEPTH_GATES.md`; runs `runs/dyck_mdepth`. Primary metric A2f (Hewitt distance-averaged closing accuracy over prefixes where the sampler can close; chance 0.500; CE-optimal predictor 1.000). Paired by seed, n=8, MDE exact-t (house 2.8 beside).

**Reproduction (in batch, ladder defaults L32 D4):** RoPE-4L: 0 weights differ, eval grid equal True, loss curve max diff 4.8e-08; MapWM-4L_r2: 0 weights differ, eval grid equal True, loss curve max diff 8.1e-08

## PRIMARY -- 4 layers, 3x budget, trained at L32 D12, scored at L32 D12 (A2f)

- index base 10000 (as the ladder): position main +0.003 (MDE 0.000 exact-t / 0.000 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- index base 32: position main +0.003 (MDE 0.001 exact-t / 0.001 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- registered effective (stronger index base per encoding: RoPE_b32, PoPE): **+0.002 (MDE 0.000 exact-t / 0.000 house, 8/8 +, sign-flip p 0.0078) DETECTABLE**
- best A2f floor at L32 D12 (gates, T12 fit): 0.593518407431268

**Verdict: CLOSES: at matched depth the position effect at L32 D12 is below 0.05 A2f or within its MDE -- the ladder's +0.168 was depth EXTRAPOLATION; the language line has no matched-distribution positive result beyond the shrinking D4 training-cell effect.**

## What this means

**The 4-layer headline closes.** Trained at depth 12, every 4-layer arm is at ceiling on the cell
the ladder reported: index 0.997-0.998, path-integrated 1.000, against a 0.594 floor. The +0.002
position effect is 8/8 and clears a vanishing MDE, but it is a ceiling difference, and 0.002 is
nowhere near the 0.05 floor the pre-registration set. **The ladder's +0.168 at L32 D12 was
extrapolation to nesting depth the models never trained on.**

**What survives, at matched depth: depth substitutes for path integration, and that is the whole
story.** At the 1x budget the matched-depth ladder gives +0.353 / +0.130 / +0.045 / +0.024 at 1-4
layers (8/8 each). One layer of path integration is worth roughly three layers of attention here.
That is a real effect at matched distribution, but it is a claim about parameter efficiency, not
about something depth cannot buy -- and the old "depth closes 40% of the gap, then stops" reading
is dead: index arms keep climbing (3L -> 4L +0.018 / +0.025) and reach ceiling by 4 layers at 3x.

**Two confounds closed in the same batch.**
- *Budget*: the 1x ladder budget IS limiting the index arms (3x - 1x = +0.021 index mean, 8/8),
  which is part of why they looked plateaued.
- *Frequency ladder*: index base 32 minus base 10000 is +0.001 / -0.001, both within MDE. The
  `base=10000` confound flagged 2026-09-21 is settled at 4 layers, not just at 1.

**Mixture training (depth drawn from 4..12) keeps a real effect**: +0.110 at L32 D12 (8/8) and
+0.043 at D4, with L128 D12 unmeasured. So the effect survives when depth VARIES in training; what
kills it is training at a single fixed depth equal to the test depth, with enough budget and layers.

## Arm means at L32 D12 (A2f / plain A2)

| condition | depth | RoPE | PoPE | RoPE_b32 | PoPE_b32 | MapWM | MapPoPE |
|---|---|---|---|---|---|---|---|
| T12x3 | 4L | 0.998 / 0.980 | 0.997 / 0.985 | 0.998 / 0.980 | 0.997 / 0.988 | 1.000 / 0.982 | 1.000 / 0.983 |
| T12 | 1L | 0.648 / 0.647 | 0.640 / 0.640 | -- | -- | 0.996 / 0.991 | 0.998 / 0.994 |
| T12 | 2L | 0.889 / 0.881 | 0.850 / 0.845 | -- | -- | 1.000 / 0.988 | 1.000 / 0.994 |
| T12 | 3L | 0.964 / 0.951 | 0.946 / 0.937 | -- | -- | 1.000 / 0.987 | 1.000 / 0.989 |
| T12 | 4L | 0.982 / 0.967 | 0.971 / 0.960 | -- | -- | 1.000 / 0.985 | 1.000 / 0.985 |
| Tmix | 4L | 0.898 / 0.899 | 0.881 / 0.881 | -- | -- | 0.999 / 0.999 | 0.999 / 0.999 |
| ladder_T4 | 1L | 0.655 / 0.657 | 0.665 / 0.667 | -- | -- | 0.914 / 0.915 | 0.988 / 0.988 |
| ladder_T4 | 2L | 0.737 / 0.740 | 0.763 / 0.765 | -- | -- | 0.936 / 0.937 | 0.987 / 0.987 |
| ladder_T4 | 3L | 0.757 / 0.760 | 0.774 / 0.777 | -- | -- | 0.932 / 0.934 | 0.919 / 0.920 |
| ladder_T4 | 4L | 0.752 / 0.755 | 0.778 / 0.781 | -- | -- | 0.922 / 0.923 | 0.948 / 0.949 |

`ladder_T4` = the D4-trained ladder checkpoints re-scored on the same sequences (cross-batch reference: the depth-OOD effect).

## Secondaries

S1, depth ladder at matched depth (T12, 1x budget), position main at L32 D12:
- 1L: +0.353 (MDE 0.003 exact-t / 0.002 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 2L: +0.130 (MDE 0.025 exact-t / 0.021 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 3L: +0.045 (MDE 0.009 exact-t / 0.008 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 4L: +0.024 (MDE 0.007 exact-t / 0.006 house, 8/8 +, sign-flip p 0.0078) DETECTABLE

S1, index arms 3L -> 4L (plateau check, T12 1x):
- RoPE: +0.018 (MDE 0.022 exact-t / 0.019 house, 7/8 +, sign-flip p 0.0391) within MDE
- PoPE: +0.025 (MDE 0.015 exact-t / 0.013 house, 7/8 +, sign-flip p 0.0156) DETECTABLE

S2, mixture training D in {4..12}, 4 layers, position main:
- L32D4: +0.043 (MDE 0.004 exact-t / 0.003 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- L32D12: +0.110 (MDE 0.016 exact-t / 0.013 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- L128D12: +0.023 (MDE 0.089 exact-t / 0.076 house, 4/8 +, sign-flip p 0.4766) within MDE

S3, index base 32 minus base 10000 (4L, 3x, L32 D12):
- RoPE: +0.001 (MDE 0.001 exact-t / 0.001 house, 5/8 +, sign-flip p 0.2031) within MDE
- PoPE: -0.001 (MDE 0.001 exact-t / 0.001 house, 3/8 +, sign-flip p 0.1172) within MDE

S4, budget 3x minus 1x at 4L (T12, L32 D12):
- RoPE: +0.016 (MDE 0.012 exact-t / 0.010 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- PoPE: +0.026 (MDE 0.010 exact-t / 0.009 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- MapWM: +0.000 (MDE 0.000 exact-t / 0.000 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- MapPoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- index mean: +0.021 (MDE 0.007 exact-t / 0.006 house, 8/8 +, sign-flip p 0.0078) DETECTABLE -> 1x T12 ladder budget-limited: True

Reference, the ladder's D4-trained position main at L32 D12 re-scored with A2f:
- 1L: +0.291 (MDE 0.038 exact-t / 0.033 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 2L: +0.211 (MDE 0.011 exact-t / 0.010 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 3L: +0.160 (MDE 0.042 exact-t / 0.036 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 4L: +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE

Reference, length extrapolation L128 D12 (4L):
- T12x3: +0.079 (MDE 0.051 exact-t / 0.044 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- T12: +0.071 (MDE 0.048 exact-t / 0.041 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- ladder_T4: +0.002 (MDE 0.035 exact-t / 0.030 house, 4/8 +, sign-flip p 0.8359) within MDE

## Convergence (final loss = mean of the last 10% of the curve; floor = the training distribution's sampler entropy)

| condition / arm / depth | final loss | gap to floor (mean / max) | median slope /1k |
|---|---|---|---|
| T12x3|RoPE|4L | 0.5280 | +0.0082 / +0.0105 | -0.00007 |
| T12x3|PoPE|4L | 0.5308 | +0.0110 / +0.0124 | -0.00016 |
| T12x3|RoPE_b32|4L | 0.5266 | +0.0068 / +0.0083 | +0.00002 |
| T12x3|PoPE_b32|4L | 0.5345 | +0.0146 / +0.0165 | -0.00027 |
| T12x3|MapWM|4L | 0.5204 | +0.0006 / +0.0010 | +0.00003 |
| T12x3|MapPoPE|4L | 0.5210 | +0.0011 / +0.0023 | -0.00003 |
| T12|RoPE|1L | 0.7688 | +0.2490 / +0.2556 | -0.00051 |
| T12|PoPE|1L | 0.7941 | +0.2743 / +0.3222 | -0.00148 |
| T12|MapWM|1L | 0.5617 | +0.0418 / +0.0813 | -0.00094 |
| T12|MapPoPE|1L | 0.6207 | +0.1009 / +0.1659 | -0.00129 |
| T12|RoPE|2L | 0.6228 | +0.1030 / +0.1230 | +0.00016 |
| T12|PoPE|2L | 0.6538 | +0.1339 / +0.1584 | -0.00231 |
| T12|MapWM|2L | 0.5305 | +0.0107 / +0.0250 | +0.00005 |
| T12|MapPoPE|2L | 0.5386 | +0.0188 / +0.0382 | -0.00269 |
| T12|RoPE|3L | 0.5690 | +0.0492 / +0.0542 | -0.00124 |
| T12|PoPE|3L | 0.5848 | +0.0650 / +0.0775 | -0.00212 |
| T12|MapWM|3L | 0.5291 | +0.0093 / +0.0172 | -0.00025 |
| T12|MapPoPE|3L | 0.5266 | +0.0067 / +0.0126 | +0.00003 |
| T12|RoPE|4L | 0.5508 | +0.0310 / +0.0397 | -0.00095 |
| T12|PoPE|4L | 0.5625 | +0.0427 / +0.0531 | -0.00131 |
| T12|MapWM|4L | 0.5235 | +0.0037 / +0.0059 | +0.00038 |
| T12|MapPoPE|4L | 0.5258 | +0.0059 / +0.0129 | +0.00037 |
| Tmix|RoPE|4L | 0.8279 | +0.1468 / +0.1549 | +0.00489 |
| Tmix|PoPE|4L | 0.8350 | +0.1539 / +0.1594 | +0.00474 |
| Tmix|MapWM|4L | 0.7734 | +0.0922 / +0.1107 | +0.01172 |
| Tmix|MapPoPE|4L | 0.7924 | +0.1113 / +0.1332 | +0.01055 |
| ladder_T4|RoPE|1L | 0.9749 | +0.1744 / +0.1775 | -0.00176 |
| ladder_T4|PoPE|1L | 0.9869 | +0.1864 / +0.1901 | -0.00122 |
| ladder_T4|MapWM|1L | 0.8483 | +0.0478 / +0.0592 | -0.00109 |
| ladder_T4|MapPoPE|1L | 0.8498 | +0.0493 / +0.0608 | -0.00200 |
| ladder_T4|RoPE|2L | 0.8853 | +0.0848 / +0.0892 | -0.00211 |
| ladder_T4|PoPE|2L | 0.8915 | +0.0910 / +0.0931 | -0.00144 |
| ladder_T4|MapWM|2L | 0.8247 | +0.0242 / +0.0345 | -0.00194 |
| ladder_T4|MapPoPE|2L | 0.8319 | +0.0314 / +0.0384 | -0.00276 |
| ladder_T4|RoPE|3L | 0.8524 | +0.0519 / +0.0564 | -0.00216 |
| ladder_T4|PoPE|3L | 0.8583 | +0.0578 / +0.0595 | -0.00201 |
| ladder_T4|MapWM|3L | 0.8164 | +0.0159 / +0.0322 | -0.00224 |
| ladder_T4|MapPoPE|3L | 0.8291 | +0.0286 / +0.0385 | -0.00230 |
| ladder_T4|RoPE|4L | 0.8316 | +0.0311 / +0.0341 | -0.00224 |
| ladder_T4|PoPE|4L | 0.8378 | +0.0373 / +0.0392 | -0.00215 |
| ladder_T4|MapWM|4L | 0.8130 | +0.0125 / +0.0277 | -0.00217 |
| ladder_T4|MapPoPE|4L | 0.8227 | +0.0222 / +0.0317 | -0.00243 |
