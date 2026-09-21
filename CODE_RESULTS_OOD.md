# Code, beyond the training context: MDEs

Trained at seq 512, evaluated at 2048. Seeds [0, 1, 2]. Pre-registration: `CODE_PREREG.md` Amendment 1.

## O-A  val bpc by position (lower better), seed means

| arm | 0-511 | 512-1023 | 1024-2047 |
|---|---|---|---|
| RoPE (index/RoPE) | 0.8755 | 2.6713 | 4.4641 |
| PoPE (index/PoPE) | 0.8785 | 0.7991 | 0.8792 |
| MapWM (path/RoPE) | 0.8834 | 1.6070 | 4.5815 |
| MapPoPE (path/PoPE) | 0.8828 | 0.7587 | 0.7776 |

**Position 0-511**
- **O2  MapPoPE - PoPE   (composition: beats its ENCODING component)**: +0.0043 (MDE 0.0074, better 1/3) unmeasured
- **O2  MapPoPE - MapWM  (composition: beats its POSITION component)**: -0.0006 (MDE 0.0099, better 1/3) unmeasured
- **O1  MapWM - RoPE     (path integration on the RoPE row)**: +0.0079 (MDE 0.0061, better 0/3) **DETECTABLE**
- **    PoPE - RoPE      (encoding on the index row)**: +0.0029 (MDE 0.0084, better 1/3) unmeasured

**Position 512-1023**
- **O2  MapPoPE - PoPE   (composition: beats its ENCODING component)**: -0.0404 (MDE 0.0429, better 3/3) unmeasured
- **O2  MapPoPE - MapWM  (composition: beats its POSITION component)**: -0.8484 (MDE 0.5991, better 3/3) **DETECTABLE**
- **O1  MapWM - RoPE     (path integration on the RoPE row)**: -1.0643 (MDE 0.4445, better 3/3) **DETECTABLE**
- **    PoPE - RoPE      (encoding on the index row)**: -1.8722 (MDE 0.1406, better 3/3) **DETECTABLE**

**Position 1024-2047**
- **O2  MapPoPE - PoPE   (composition: beats its ENCODING component)**: -0.1016 (MDE 0.0521, better 3/3) **DETECTABLE**
- **O2  MapPoPE - MapWM  (composition: beats its POSITION component)**: -3.8039 (MDE 0.3822, better 3/3) **DETECTABLE**
- **O1  MapWM - RoPE     (path integration on the RoPE row)**: +0.1175 (MDE 0.5288, better 1/3) unmeasured
- **    PoPE - RoPE      (encoding on the index row)**: -3.5849 (MDE 0.1659, better 3/3) **DETECTABLE**

## O-B  closer accuracy by bracket distance (higher better), seed means

| arm | 0-32 | 33-128 | 129-512 | 513-1024 |
|---|---|---|---|---|
| RoPE (index/RoPE) | 0.927 | 0.861 | 0.738 | 0.550 |
| PoPE (index/PoPE) | 0.999 | 0.990 | 0.946 | 0.854 |
| MapWM (path/RoPE) | 0.909 | 0.868 | 0.801 | 0.650 |
| MapPoPE (path/PoPE) | 0.999 | 0.986 | 0.919 | 0.848 |
| *no-stack floor* | *0.890* | *0.756* | *0.758* | *0.675* |

**Distance 0-32** (floor 0.890)
- **MapPoPE - PoPE**: -0.0001 (MDE 0.0009, better 1/3) unmeasured
- **MapPoPE - MapWM**: +0.0903 (MDE 0.0535, better 3/3) **DETECTABLE**
- **MapWM - RoPE**: -0.0179 (MDE 0.0676, better 1/3) unmeasured

**Distance 33-128** (floor 0.756)
- **MapPoPE - PoPE**: -0.0045 (MDE 0.0143, better 1/3) unmeasured
- **MapPoPE - MapWM**: +0.1171 (MDE 0.0350, better 3/3) **DETECTABLE**
- **MapWM - RoPE**: +0.0070 (MDE 0.0702, better 2/3) unmeasured

**Distance 129-512** (floor 0.758)
- **MapPoPE - PoPE**: -0.0270 (MDE 0.0341, better 0/3) unmeasured
- **MapPoPE - MapWM**: +0.1180 (MDE 0.0478, better 3/3) **DETECTABLE**
- **MapWM - RoPE**: +0.0632 (MDE 0.0474, better 3/3) **DETECTABLE**

**Distance 513-1024** (floor 0.675)
- **MapPoPE - PoPE**: -0.0054 (MDE 0.0648, better 1/3) unmeasured
- **MapPoPE - MapWM**: +0.1978 (MDE 0.0076, better 3/3) **DETECTABLE**
- **MapWM - RoPE**: +0.1003 (MDE 0.1094, better 3/3) unmeasured

## Rule 9 and loss overlap

- r(final train bpc, OOD bpc at 1024-2047) = **-0.097** over 12 runs.
  - Below 0.98, so the OOD effect is not simply a convergence gap.
- final train bpc ranges: RoPE (index/RoPE) [0.8502, 0.8814]; PoPE (index/PoPE) [0.8702, 0.9243]; MapWM (path/RoPE) [0.8601, 0.9948]; MapPoPE (path/PoPE) [0.8452, 0.9638]
- MapPoPE vs PoPE training losses overlap: **True**

