# Dyck position effect at MATCHED nesting depth -- results

Pre-registration `DYCK_MDEPTH_PREREG.md`; gates `DYCK_MDEPTH_GATES.md`; runs `runs/dyck_mdepth`. Primary metric A2f (Hewitt distance-averaged closing accuracy over prefixes where the sampler can close; chance 0.500; CE-optimal predictor 1.000). Paired by seed, n=8, MDE exact-t (house 2.8 beside).

**Reproduction (in batch, ladder defaults L32 D4):** RoPE-4L: 0 weights differ, eval grid equal True, loss curve max diff 0.0e+00; MapWM-4L_r2: 0 weights differ, eval grid equal True, loss curve max diff 0.0e+00

## PRIMARY -- 4 layers, 3x budget, trained at L32 D12, scored at L32 D12 (A2f)

- index base 10000 (as the ladder): position main +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- index base 32: position main +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- registered effective (stronger index base per encoding: RoPE, PoPE): **+0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE**
- best A2f floor at L32 D12 (gates, T12 fit): 0.593518407431268

**Verdict: SURVIVES: path integration beats the index arms at MATCHED depth (L32 D12 in the training distribution), 4 layers, 3x budget, by >= 0.10 A2f -- a matched-distribution capability result.**

## Arm means at L32 D12 (A2f / plain A2)

| condition | depth | RoPE | PoPE | RoPE_b32 | PoPE_b32 | MapWM | MapPoPE |
|---|---|---|---|---|---|---|---|
| T12x3 | 4L | 0.752 / 0.755 | 0.778 / 0.781 | 0.752 / 0.755 | 0.778 / 0.781 | 0.922 / 0.923 | 0.948 / 0.949 |
| T12 | 1L | 0.655 / 0.657 | 0.665 / 0.667 | -- | -- | 0.914 / 0.915 | 0.988 / 0.988 |
| T12 | 2L | 0.737 / 0.740 | 0.763 / 0.765 | -- | -- | 0.936 / 0.937 | 0.987 / 0.987 |
| T12 | 3L | 0.757 / 0.760 | 0.774 / 0.777 | -- | -- | 0.932 / 0.934 | 0.919 / 0.920 |
| T12 | 4L | 0.752 / 0.755 | 0.778 / 0.781 | -- | -- | 0.922 / 0.923 | 0.948 / 0.949 |
| Tmix | 4L | 0.752 / 0.755 | 0.778 / 0.781 | -- | -- | 0.922 / 0.923 | 0.948 / 0.949 |
| ladder_T4 | 1L | 0.655 / 0.657 | 0.665 / 0.667 | -- | -- | 0.914 / 0.915 | 0.988 / 0.988 |
| ladder_T4 | 2L | 0.737 / 0.740 | 0.763 / 0.765 | -- | -- | 0.936 / 0.937 | 0.987 / 0.987 |
| ladder_T4 | 3L | 0.757 / 0.760 | 0.774 / 0.777 | -- | -- | 0.932 / 0.934 | 0.919 / 0.920 |
| ladder_T4 | 4L | 0.752 / 0.755 | 0.778 / 0.781 | -- | -- | 0.922 / 0.923 | 0.948 / 0.949 |

`ladder_T4` = the D4-trained ladder checkpoints re-scored on the same sequences (cross-batch reference: the depth-OOD effect).

## Secondaries

S1, depth ladder at matched depth (T12, 1x budget), position main at L32 D12:
- 1L: +0.291 (MDE 0.038 exact-t / 0.033 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 2L: +0.211 (MDE 0.011 exact-t / 0.010 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 3L: +0.160 (MDE 0.042 exact-t / 0.036 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 4L: +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE

S1, index arms 3L -> 4L (plateau check, T12 1x):
- RoPE: -0.005 (MDE 0.017 exact-t / 0.014 house, 2/8 +, sign-flip p 0.3750) within MDE
- PoPE: +0.003 (MDE 0.009 exact-t / 0.008 house, 4/8 +, sign-flip p 0.3047) within MDE

S2, mixture training D in {4..12}, 4 layers, position main:
- L32D4: +0.019 (MDE 0.004 exact-t / 0.003 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- L32D12: +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- L128D12: +0.002 (MDE 0.035 exact-t / 0.030 house, 4/8 +, sign-flip p 0.8359) within MDE

S3, index base 32 minus base 10000 (4L, 3x, L32 D12):
- RoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE
- PoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE

S4, budget 3x minus 1x at 4L (T12, L32 D12):
- RoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE
- PoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE
- MapWM: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE
- MapPoPE: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE
- index mean: +0.000 (MDE 0.000 exact-t / 0.000 house, 0/8 +, sign-flip p 1.0000) within MDE -> 1x T12 ladder budget-limited: False

Reference, the ladder's D4-trained position main at L32 D12 re-scored with A2f:
- 1L: +0.291 (MDE 0.038 exact-t / 0.033 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 2L: +0.211 (MDE 0.011 exact-t / 0.010 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 3L: +0.160 (MDE 0.042 exact-t / 0.036 house, 8/8 +, sign-flip p 0.0078) DETECTABLE
- 4L: +0.170 (MDE 0.029 exact-t / 0.025 house, 8/8 +, sign-flip p 0.0078) DETECTABLE

Reference, length extrapolation L128 D12 (4L):
- T12x3: +0.002 (MDE 0.035 exact-t / 0.030 house, 4/8 +, sign-flip p 0.8359) within MDE
- T12: +0.002 (MDE 0.035 exact-t / 0.030 house, 4/8 +, sign-flip p 0.8359) within MDE
- ladder_T4: +0.002 (MDE 0.035 exact-t / 0.030 house, 4/8 +, sign-flip p 0.8359) within MDE

## Convergence (final loss = mean of the last 10% of the curve; floor = the training distribution's sampler entropy)

| condition / arm / depth | final loss | gap to floor (mean / max) | median slope /1k |
|---|---|---|---|
| T12x3|RoPE|4L | 0.8316 | +0.3116 / +0.3146 | -0.00224 |
| T12x3|PoPE|4L | 0.8378 | +0.3178 / +0.3197 | -0.00215 |
| T12x3|RoPE_b32|4L | 0.8316 | +0.3116 / +0.3146 | -0.00224 |
| T12x3|PoPE_b32|4L | 0.8378 | +0.3178 / +0.3197 | -0.00215 |
| T12x3|MapWM|4L | 0.8130 | +0.2930 / +0.3082 | -0.00217 |
| T12x3|MapPoPE|4L | 0.8227 | +0.3027 / +0.3122 | -0.00243 |
| T12|RoPE|1L | 0.9749 | +0.4549 / +0.4580 | -0.00176 |
| T12|PoPE|1L | 0.9869 | +0.4669 / +0.4706 | -0.00122 |
| T12|MapWM|1L | 0.8483 | +0.3283 / +0.3397 | -0.00109 |
| T12|MapPoPE|1L | 0.8498 | +0.3298 / +0.3413 | -0.00200 |
| T12|RoPE|2L | 0.8853 | +0.3653 / +0.3697 | -0.00211 |
| T12|PoPE|2L | 0.8915 | +0.3715 / +0.3736 | -0.00144 |
| T12|MapWM|2L | 0.8247 | +0.3047 / +0.3150 | -0.00194 |
| T12|MapPoPE|2L | 0.8319 | +0.3119 / +0.3189 | -0.00276 |
| T12|RoPE|3L | 0.8524 | +0.3324 / +0.3369 | -0.00216 |
| T12|PoPE|3L | 0.8583 | +0.3383 / +0.3400 | -0.00201 |
| T12|MapWM|3L | 0.8164 | +0.2964 / +0.3127 | -0.00224 |
| T12|MapPoPE|3L | 0.8291 | +0.3091 / +0.3190 | -0.00230 |
| T12|RoPE|4L | 0.8316 | +0.3116 / +0.3146 | -0.00224 |
| T12|PoPE|4L | 0.8378 | +0.3178 / +0.3197 | -0.00215 |
| T12|MapWM|4L | 0.8130 | +0.2930 / +0.3082 | -0.00217 |
| T12|MapPoPE|4L | 0.8227 | +0.3027 / +0.3122 | -0.00243 |
| Tmix|RoPE|4L | 0.8316 | +0.3116 / +0.3146 | -0.00224 |
| Tmix|PoPE|4L | 0.8378 | +0.3178 / +0.3197 | -0.00215 |
| Tmix|MapWM|4L | 0.8130 | +0.2930 / +0.3082 | -0.00217 |
| Tmix|MapPoPE|4L | 0.8227 | +0.3027 / +0.3122 | -0.00243 |
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
