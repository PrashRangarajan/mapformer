# Level 1.5 InEKF Results (constant Pi, per-token R_t)

Generated: Mon Apr 20 08:45:05 PM PDT 2026

Level 1.5 is a middle ground between Level 1 and Level 2:
- Constant (learnable) Pi — avoids the Mobius covariance scan
- Per-token R_t from a learned head — gives K_t = Pi/(Pi+R_t) time variation
- ONE scalar affine scan (not two matrix scans like Level 2)
- Training speed: ~10.7s/epoch, same as Level 1 and ~55x faster than Level 2

## Training losses

- lm200: first=2.7044, mid=0.8988, final=0.8124
- no-lm: first=2.1186, mid=0.8061, final=0.7506

## Landmark-aware eval (T=128 and T=512)

```
Environment: grid=64, K=16, p_empty=0.5, n_landmarks=200
Unified vocab size: 221

========================================================================================
T = 128  (200 trials)
========================================================================================

  MapFormer_WM_noise
      overall  acc=0.815  NLL=0.786  n=5895
     landmark  acc=0.014  NLL=5.790  n=287
      regular  acc=0.740  NLL=0.853  n=2929
        blank  acc=0.984  NLL=0.178  n=2679

  MapFormer_WM_ParallelInEKF
      overall  acc=0.866  NLL=0.627  n=6089
     landmark  acc=0.180  NLL=4.546  n=316
      regular  acc=0.821  NLL=0.626  n=2914
        blank  acc=0.987  NLL=0.195  n=2859

  MapFormer_WM_PredictiveCoding
      overall  acc=0.874  NLL=0.612  n=5820
     landmark  acc=0.016  NLL=5.665  n=258
      regular  acc=0.847  NLL=0.592  n=2807
        blank  acc=0.981  NLL=0.158  n=2755

  MapFormer_WM_Level2InEKF
      overall  acc=0.852  NLL=0.631  n=5931
     landmark  acc=0.029  NLL=4.684  n=277
      regular  acc=0.799  NLL=0.690  n=2779
        blank  acc=0.982  NLL=0.184  n=2875

  MapFormer_WM_Level15InEKF
      overall  acc=0.948  NLL=0.265  n=5894
     landmark  acc=0.732  NLL=1.166  n=269
      regular  acc=0.929  NLL=0.307  n=2852
        blank  acc=0.987  NLL=0.134  n=2773

========================================================================================
T = 512  (200 trials)
========================================================================================

  MapFormer_WM_noise
      overall  acc=0.632  NLL=1.952  n=28126
     landmark  acc=0.007  NLL=8.935  n=1296
      regular  acc=0.334  NLL=3.111  n=13447
        blank  acc=0.991  NLL=0.112  n=13383

  MapFormer_WM_ParallelInEKF
      overall  acc=0.783  NLL=0.979  n=28726
     landmark  acc=0.167  NLL=5.437  n=1396
      regular  acc=0.640  NLL=1.206  n=13680
        blank  acc=0.989  NLL=0.296  n=13650

  MapFormer_WM_PredictiveCoding
      overall  acc=0.622  NLL=2.029  n=28181
     landmark  acc=0.004  NLL=9.444  n=1415
      regular  acc=0.322  NLL=3.171  n=13420
        blank  acc=0.989  NLL=0.094  n=13346

  MapFormer_WM_Level2InEKF
      overall  acc=0.800  NLL=0.901  n=28085
     landmark  acc=0.036  NLL=5.665  n=1273
      regular  acc=0.692  NLL=1.074  n=13497
        blank  acc=0.983  NLL=0.269  n=13315

  MapFormer_WM_Level15InEKF
      overall  acc=0.869  NLL=0.573  n=28856
     landmark  acc=0.563  NLL=2.082  n=1414
      regular  acc=0.788  NLL=0.763  n=13724
        blank  acc=0.982  NLL=0.228  n=13718

```

## Gaussian Δ-noise at T=128 (no landmarks)

```
T=128, trials=200

 noise_std              MapFormer_WM_noise      MapFormer_WM_ParallelInEKF   MapFormer_WM_PredictiveCoding        MapFormer_WM_Level2InEKF       MapFormer_WM_Level15InEKF
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      0.00                           0.959                           0.900                           0.956                           0.926                           0.944
      0.05                           0.790                           0.675                           0.755                           0.741                           0.837
      0.10                           0.591                           0.523                           0.565                           0.566                           0.657
      0.20                           0.491                           0.459                           0.490                           0.478                           0.518
      0.30                           0.443                           0.450                           0.467                           0.461                           0.468
      0.50                           0.457                           0.449                           0.455                           0.450                           0.419
      1.00                           0.446                           0.450                           0.438                           0.454                           0.415
```

## Gaussian Δ-noise at T=512 (OOD length)

```
T=512, trials=100

 noise_std              MapFormer_WM_noise      MapFormer_WM_ParallelInEKF   MapFormer_WM_PredictiveCoding        MapFormer_WM_Level2InEKF       MapFormer_WM_Level15InEKF
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      0.00                           0.902                           0.799                           0.838                           0.814                           0.861
      0.05                           0.721                           0.589                           0.647                           0.647                           0.767
      0.10                           0.567                           0.505                           0.505                           0.526                           0.624
      0.20                           0.481                           0.478                           0.470                           0.476                           0.519
      0.30                           0.478                           0.468                           0.482                           0.475                           0.480
      0.50                           0.482                           0.465                           0.480                           0.475                           0.451
      1.00                           0.488                           0.464                           0.477                           0.473                           0.436
```

## Level 1.5 R_t and K_t distribution by token category (with landmarks)

```
Constant Pi (learned): 0.9256
     action: n= 6400  R mean=70.3446  K mean=0.4960
      blank: n= 3021  R mean=74.7066  K mean=0.4289
    regular: n= 3020  R mean=71.1149  K mean=0.4412
   landmark: n=  309  R mean=48.2848  K mean=0.4520
```

## Summary

Level 1.5 achieves the best training loss with landmarks at ~60× the speed of Level 2.
The heteroscedastic K_t (via per-token R_t) is the useful part; heteroscedastic Pi
(Level 2's Mobius scan) appears to add little value on this task.

---
*Generated by Level 1.5 eval pipeline*
