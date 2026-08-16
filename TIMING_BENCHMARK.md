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
