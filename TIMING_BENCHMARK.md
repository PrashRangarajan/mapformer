> **RE-MEASURED 2026-08-09 with `torch.no_grad()` on forward-only.** The first
> run timed autograd graph construction inside its "forward only" rows, a cost
> scaling with node count rather than FLOPs, which penalised the Python-loop
> models far more than the cumsum ones -- biasing the comparison **toward the
> parallel-scan claim**. 15 reps, IQR retained (typically <0.5% of the median,
> so these numbers are not noise-limited).

# Wall-clock scaling with sequence length

batch=4, d_model=128, n_layers=2, median of 15 reps after 3 warmups, `torch.cuda.synchronize()` around every timed region; IQR over reps stored in the JSON. Forward-only runs under `torch.no_grad()` -- without it the measurement also times autograd graph construction, which scales with node count and penalises the Python-loop models.

**These models are NOT parameter-matched, so absolute times are not an architecture comparison.** What is comparable is each model's SCALING with L — the O(log T) parallel-scan claim.

## Forward + backward (training cost)

| variant | params | L=128 | L=256 | L=512 | L=1024 | L=2048 | growth (span shown) |
|---|---|---|---|---|---|---|---|
| Vanilla | 405,472 | 4.8 | 5.9 | 6.2 | 5.8 | 12.1 | **2.5x (2048/128)** |
| VanillaEM_P0 | 405,600 | 4.3 | 4.5 | 4.7 | 4.8 | 16.7 | **3.9x (2048/128)** |
| PlainFlat | 405,024 | 4.2 | 4.1 | 4.1 | 4.2 | 11.4 | **2.7x (2048/128)** |
| MapEM_NC_L | 430,016 | 28.5 | 57.9 | 110.8 | 217.5 | 410.1 | **14.4x (2048/128)** |
| TEMFaithful | 20,705 | 163.2 | 465.7 | 1499.1 | 5273.1 | 19646.3 | **120.4x (2048/128)** |

## Forward only

| variant | L=128 | L=256 | L=512 | L=1024 | L=2048 |
|---|---|---|---|---|---|
| Vanilla | 1.2 | 1.2 | 1.2 | 1.4 | 4.7 |
| VanillaEM_P0 | 1.1 | 1.3 | 1.1 | 1.7 | 6.5 |
| PlainFlat | 1.3 | 1.3 | 1.3 | 1.4 | 4.6 |
| MapEM_NC_L | 6.7 | 11.9 | 21.9 | 43.0 | 88.3 |
| TEMFaithful | 40.5 | 90.4 | 218.3 | 587.6 | 1803.3 |


## What the correction changed

**Forward+backward: unchanged, and it is the headline.** Backward legitimately
requires the graph, so this table was never affected. It reproduces to within
run-to-run variance (2.5x / 3.9x / 2.7x / 14.4x / 120.4x vs 2.9 / 3.3 / 2.6 /
14.5 / 120.2 before), and the absolute gaps at L=2048 stand: Vanilla is **34x**
faster than MapEM-NC and **1624x** faster than TEMFaithful.

**Forward-only: I was measuring my own overhead, and it flattered my conclusion.**

| @ L=2048 | biased | corrected | change |
|---|---|---|---|
| Vanilla | 4.9 | 4.7 | -4% |
| PlainFlat | 4.7 | 4.6 | -2% |
| **MapEM_NC_L** | 132.0 | **88.3** | **-33%** |
| **TEMFaithful** | 2343.9 | **1803.3** | **-23%** |

The parallel models barely moved; the Python-loop models got much faster once
they stopped being charged for graph nodes they would never build at inference.
Corrected inference-time gaps:

| | as published | corrected |
|---|---|---|
| Vanilla vs MapEM-NC | 26.9x | **18.8x** |
| Vanilla vs TEMFaithful | 478x | **384x** |

The qualitative claim survives -- forward-only scaling is 3.5-3.9x for the
parallel models against 13.2x (MapEM-NC) and 44.5x (TEMFaithful) -- but the
inference-time advantage was overstated by roughly a third, in the direction that
favoured the result. Cite the corrected numbers.
