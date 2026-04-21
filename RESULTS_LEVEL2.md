# Level 2 InEKF Results (heteroscedastic R_t)

Generated: Mon Apr 20 08:12:51 PM PDT 2026

Checkpoints:
- With landmarks: `mapformer/figures_inekf_level2_lm200/MapFormer_WM_Level2InEKF.pt` (1.2MiB)
- No landmarks:   `mapformer/figures_inekf_level2/MapFormer_WM_Level2InEKF.pt` (1006KiB)

## 1. Training losses

- lm200: first=2.7050, mid=1.2025, final=1.1334
- no-lm: first=2.1154, mid=0.8110, final=0.7678

## 2. Landmark-aware eval (Level 2 vs Level 1 vs Vanilla, with 200 landmarks)

```
Environment: grid=64, K=16, p_empty=0.5, n_landmarks=200
Unified vocab size: 221

========================================================================================
T = 128  (200 trials)
========================================================================================

  MapFormer_WM_noise
      overall  acc=0.822  NLL=0.757  n=5820
     landmark  acc=0.026  NLL=5.732  n=272
      regular  acc=0.741  NLL=0.844  n=2818
        blank  acc=0.985  NLL=0.172  n=2730

  MapFormer_WM_ParallelInEKF
      overall  acc=0.851  NLL=0.660  n=5886
     landmark  acc=0.165  NLL=4.463  n=285
      regular  acc=0.795  NLL=0.707  n=2884
        blank  acc=0.983  NLL=0.210  n=2717

  MapFormer_WM_PredictiveCoding
      overall  acc=0.860  NLL=0.658  n=5605
     landmark  acc=0.023  NLL=5.508  n=299
      regular  acc=0.836  NLL=0.597  n=2691
        blank  acc=0.980  NLL=0.166  n=2615

  MapFormer_WM_Level2InEKF
      overall  acc=0.868  NLL=0.605  n=5897
     landmark  acc=0.055  NLL=4.738  n=343
      regular  acc=0.852  NLL=0.536  n=2786
        blank  acc=0.986  NLL=0.162  n=2768

========================================================================================
T = 512  (200 trials)
========================================================================================

  MapFormer_WM_noise
      overall  acc=0.630  NLL=1.924  n=28308
     landmark  acc=0.006  NLL=8.523  n=1338
      regular  acc=0.337  NLL=3.058  n=13606
        blank  acc=0.992  NLL=0.109  n=13364

  MapFormer_WM_ParallelInEKF
      overall  acc=0.785  NLL=0.995  n=28935
     landmark  acc=0.131  NLL=5.686  n=1468
      regular  acc=0.655  NLL=1.178  n=13893
        blank  acc=0.988  NLL=0.300  n=13574

  MapFormer_WM_PredictiveCoding
      overall  acc=0.624  NLL=2.015  n=28416
     landmark  acc=0.006  NLL=9.430  n=1443
      regular  acc=0.328  NLL=3.124  n=13578
        blank  acc=0.990  NLL=0.093  n=13395

  MapFormer_WM_Level2InEKF
      overall  acc=0.802  NLL=0.901  n=28162
     landmark  acc=0.037  NLL=5.649  n=1392
      regular  acc=0.697  NLL=1.054  n=13336
        blank  acc=0.986  NLL=0.256  n=13434

```

## 3. Gaussian Δ-noise robustness at T=128 (no landmarks)

```
T=128, trials=200

 noise_std              MapFormer_WM_noise      MapFormer_WM_ParallelInEKF   MapFormer_WM_PredictiveCoding        MapFormer_WM_Level2InEKF
------------------------------------------------------------------------------------------------------------------------------------------
      0.00                           0.963                           0.908                           0.954                           0.931
      0.05                           0.792                           0.665                           0.756                           0.735
      0.10                           0.598                           0.526                           0.557                           0.561
      0.20                           0.492                           0.469                           0.462                           0.473
      0.30                           0.468                           0.446                           0.458                           0.471
      0.50                           0.462                           0.437                           0.461                           0.452
      1.00                           0.459                           0.444                           0.431                           0.453
```

## 4. Gaussian Δ-noise at T=512 (4× OOD length)

```
T=512, trials=100

 noise_std              MapFormer_WM_noise      MapFormer_WM_ParallelInEKF   MapFormer_WM_PredictiveCoding        MapFormer_WM_Level2InEKF
------------------------------------------------------------------------------------------------------------------------------------------
      0.00                           0.901                           0.792                           0.839                           0.827
      0.05                           0.736                           0.601                           0.638                           0.646
      0.10                           0.551                           0.504                           0.512                           0.540
      0.20                           0.487                           0.482                           0.480                           0.492
      0.30                           0.473                           0.462                           0.467                           0.474
      0.50                           0.494                           0.476                           0.463                           0.462
      1.00                           0.485                           0.459                           0.475                           0.473
```

## 5. Clone-structure analysis (Level 2 no-landmark version)

```
============================================================

θ̂ (path/corrected):
  Mean R² across obs types: 0.270  (1.0 = perfect position recovery)
  Per-type R²:
    type     R²  n_samples   n_unique_cells
       0  0.337       1100              117
       1  0.223       1407              131
       2  0.301       1032              120
       3  0.170       1226              124
       4  0.237        964              105
       5  0.179       1228              135
       6  0.331       1572              146
       7  0.290       1129              118
       8  0.289       1335              123
       9  0.263       1307              136
      10  0.328       1176              117
      11  0.069        846              118
      12  0.226       1365              137
      13  0.209       1104              122
      14  0.336       1006              105
      15  0.064       1122              124
    BLANK (16): R²=0.295  n=19181  unique_cells=1983

hidden (pre-out):
  Mean R² across obs types: 0.274  (1.0 = perfect position recovery)
  Per-type R²:
    type     R²  n_samples   n_unique_cells
       0  0.316       1100              117
       1  0.251       1407              131
       2  0.323       1032              120
       3  0.243       1226              124
       4  0.256        964              105
       5  0.216       1228              135
       6  0.316       1572              146
       7  0.325       1129              118
       8  0.298       1335              123
       9  0.286       1307              136
      10  0.346       1176              117
      11  0.137        846              118
      12  0.307       1365              137
      13  0.327       1104              122
      14  0.297       1006              105
      15  0.221       1122              124
    BLANK (16): R²=0.266  n=19181  unique_cells=1983

============================================================
CLONE SEPARATION SCORE
============================================================
                    θ̂ — separation: 0.3908  (higher is more clone-like; 0 = no separation)
                hidden — separation: 0.1318  (higher is more clone-like; 0 = no separation)

============================================================
PCA VISUALIZATION (hidden features)
============================================================
  Obs type 6: 1572 samples, 146 unique cells
  PC1 var: 0.226  PC2 var: 0.148
  Saved: mapformer/figures_inekf_level2/clone_pca_type6.npz

Done.
```

## 6. Level 2 landmark-cell R_t and K_t distribution

Does the model learn small R_t / large K_t for landmarks?

```
     action: n= 6400  R mean=109.7727  K mean=0.1650
      blank: n= 2994  R mean=129.4192  K mean=0.1095
    regular: n= 3061  R mean=124.9393  K mean=0.0447
   landmark: n=  295  R mean=89.8406  K mean=0.0665
```

---

*Generated by rerun_level2_eval.sh (original autonomous_level2_eval.sh had a Python path bug).*
